"""End-to-end Golgi retrieval and answer generation pipeline."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from google import genai

logger = logging.getLogger(__name__)

CHUNKS_META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "chunks_metadata.json")

_SYSTEM_PROMPT = (
    "You are a healthcare information assistant. Answer the user's question using ONLY "
    "the provided context passages. Cite each passage you use with its number in brackets, "
    "e.g. [1], [2]. If the answer cannot be found in the provided context, respond with: "
    "\"I cannot find this in the retrieved sources.\" Do not use any prior knowledge."
)


class GolgiPipeline:
    """Orchestrates intent routing, hybrid retrieval, reranking, and LLM answer generation."""

    def __init__(self) -> None:
        from ingestion.chunker import CHUNKS_META_PATH as _CMP
        from orchestration.router import IntentRouter
        from reranking.cross_encoder import CrossEncoderReranker
        from retrieval.dense import DenseRetriever
        from retrieval.hybrid import HybridRetriever
        from retrieval.sparse import BM25Retriever

        self._genai = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._router = IntentRouter()
        self._reranker = CrossEncoderReranker()

        meta_path = _CMP
        with open(meta_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        self._dense = DenseRetriever()
        self._sparse = BM25Retriever(chunks)
        self._hybrid = HybridRetriever(self._dense, self._sparse, alpha=0.5)

        logger.info("GolgiPipeline initialised with %d indexed chunks", len(chunks))

    def run(self, query: str) -> dict[str, Any]:
        """Execute the full retrieval and answer pipeline.

        Args:
            query: User's natural-language question.

        Returns:
            Dict with keys: intent, answer, sources, chunks_used, retrieval_trace.
        """
        trace: list[str] = []
        t0 = time.time()

        # 1. Intent detection + source fetch
        trace.append(f"Step 1: Detecting intent for query: {query!r}")
        routed = self._router.route_and_fetch(query)
        intent = routed["intent"]
        fresh_docs = routed["docs"]
        trace.append(f"Step 2: Intent detected → {intent}. Fetched {len(fresh_docs)} live documents.")

        # 2. Chunk fresh docs and merge with indexed corpus for BM25
        if fresh_docs:
            from ingestion.chunker import chunk_documents
            fresh_chunks = chunk_documents(fresh_docs)
            trace.append(f"Step 3: Chunked {len(fresh_docs)} docs → {len(fresh_chunks)} fresh chunks.")
        else:
            fresh_chunks = []
            trace.append("Step 3: No fresh documents — using indexed corpus only.")

        # 3. Hybrid retrieval over indexed FAISS + BM25 corpus
        trace.append("Step 4: Running hybrid retrieval (dense + BM25, alpha=0.5).")
        hybrid_results = self._hybrid.search(query, k=20)

        # Augment with fresh chunks scored only by BM25 presence
        if fresh_chunks:
            from retrieval.sparse import BM25Retriever
            fresh_bm25 = BM25Retriever(fresh_chunks)
            fresh_hits = fresh_bm25.search(query, k=10)
            # Wrap as hybrid-compatible dicts with placeholder scores
            for h in fresh_hits:
                h["combined_score"] = h["score"] / (h["score"] + 1)  # squash to (0,1)
                h["dense_score"] = 0.0
                h["sparse_score"] = h["score"]
            hybrid_results = hybrid_results + fresh_hits

        trace.append(f"Step 5: Retrieved {len(hybrid_results)} candidate chunks total.")

        # 4. Cross-encoder reranking
        trace.append("Step 6: Reranking with cross-encoder (ms-marco-MiniLM-L-6-v2).")
        reranked = self._reranker.rerank(query, hybrid_results, top_k=5)
        trace.append(f"Step 7: Reranked → top {len(reranked)} chunks selected.")

        # 5. Build context for Claude
        context_parts = []
        for i, chunk in enumerate(reranked, 1):
            title = chunk["metadata"].get("title", "Untitled")
            source = chunk["metadata"].get("source", "Unknown")
            context_parts.append(f"[{i}] ({source}) {title}\n{chunk['chunk_text']}")

        context_block = "\n\n".join(context_parts)
        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

        # 6. Claude answer generation
        trace.append("Step 8: Generating answer with Gemini 2.0 Flash.")
        try:
            full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_message}"
            response = self._genai.models.generate_content(
                model="gemini-2.0-flash",
                contents=full_prompt,
            )
            answer = response.text.strip()
        except Exception as exc:
            logger.error("LLM answer generation failed: %s", exc)
            answer = "I cannot find this in the retrieved sources."

        elapsed_ms = round((time.time() - t0) * 1000)
        trace.append(f"Step 9: Done. Total time: {elapsed_ms} ms.")

        # 7. Build sources list
        sources = [
            {
                "title": c["metadata"].get("title", "Untitled"),
                "url": c["metadata"].get("url", ""),
                "source": c["metadata"].get("source", ""),
                "relevance_score": c["score"],
            }
            for c in reranked
        ]

        return {
            "intent": intent,
            "answer": answer,
            "sources": sources,
            "chunks_used": reranked,
            "retrieval_trace": trace,
        }
