"""Hybrid retriever: fuses dense (FAISS) and sparse (BM25) scores with Claude query expansion."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores to [0, 1]."""
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    """Combines dense (cosine) and sparse (BM25) retrieval with score fusion."""

    def __init__(self, dense_retriever: Any, bm25_retriever: Any, alpha: float = 0.5) -> None:
        """Initialise with pre-built retrievers.

        Args:
            dense_retriever: DenseRetriever instance.
            bm25_retriever: BM25Retriever instance.
            alpha: Weight for dense scores (1-alpha for sparse).
        """
        self._dense = dense_retriever
        self._sparse = bm25_retriever
        self._alpha = alpha

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        """Run both retrievers, fuse scores, deduplicate, and return top-k.

        Args:
            query: Free-text search query.
            k: Number of results to return.

        Returns:
            List of dicts with chunk_text, metadata, combined_score, dense_score, sparse_score.
        """
        dense_results = self._dense.search(query, k=k)
        sparse_results = self._sparse.search(query, k=k)

        dense_scores = [r["score"] for r in dense_results]
        sparse_scores = [r["score"] for r in sparse_results]

        dense_norm = _normalize(dense_scores)
        sparse_norm = _normalize(sparse_scores)

        # Build lookup by chunk_text for merging
        merged: dict[str, dict[str, Any]] = {}

        for result, norm_score in zip(dense_results, dense_norm):
            key = result["chunk_text"]
            merged[key] = {
                "chunk_text": result["chunk_text"],
                "metadata": result["metadata"],
                "dense_score": norm_score,
                "sparse_score": 0.0,
            }

        for result, norm_score in zip(sparse_results, sparse_norm):
            key = result["chunk_text"]
            if key in merged:
                merged[key]["sparse_score"] = norm_score
            else:
                merged[key] = {
                    "chunk_text": result["chunk_text"],
                    "metadata": result["metadata"],
                    "dense_score": 0.0,
                    "sparse_score": norm_score,
                }

        for entry in merged.values():
            entry["combined_score"] = round(
                self._alpha * entry["dense_score"] + (1 - self._alpha) * entry["sparse_score"],
                4,
            )

        ranked = sorted(merged.values(), key=lambda x: x["combined_score"], reverse=True)
        return ranked[:k]

    def expand_query(self, query: str) -> list[dict[str, Any]]:
        """Generate 2 alternative query phrasings via Claude, search all 3, return merged top-20.

        Args:
            query: Original search query.

        Returns:
            Merged and deduplicated top-20 results across all query variants.
        """
        from google import genai

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        try:
            prompt = (
                f"Generate exactly 2 alternative phrasings of this medical/healthcare "
                f"search query. Return only the 2 alternatives, one per line, with no "
                f"numbering, bullets, or extra text.\n\nQuery: {query}"
            )
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            raw = response.text.strip()
            alternatives = [line.strip() for line in raw.split("\n") if line.strip()][:2]
            logger.info("Query expansion generated %d alternatives", len(alternatives))
        except Exception as exc:
            logger.warning("Query expansion failed: %s — falling back to original query only", exc)
            alternatives = []

        queries = [query] + alternatives
        seen: dict[str, dict[str, Any]] = {}

        for q in queries:
            for result in self.search(q, k=20):
                key = result["chunk_text"]
                if key not in seen or result["combined_score"] > seen[key]["combined_score"]:
                    seen[key] = result

        ranked = sorted(seen.values(), key=lambda x: x["combined_score"], reverse=True)
        return ranked[:20]
