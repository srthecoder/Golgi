"""Dense retriever: loads a FAISS index and searches via cosine similarity."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "faiss.index")
CHUNKS_META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "chunks_metadata.json")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    """Cosine-similarity search over a persisted FAISS IndexFlatIP index."""

    def __init__(self, collection_name: str = "golgi") -> None:
        """Load the FAISS index and chunk metadata from disk.

        Args:
            collection_name: Unused; kept for interface compatibility with HybridRetriever.
        """
        import faiss
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(EMBED_MODEL_NAME)
        self._index = faiss.read_index(FAISS_INDEX_PATH)

        with open(CHUNKS_META_PATH, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)

        logger.info(
            "DenseRetriever: loaded FAISS index (%d vectors, dim=%d)",
            self._index.ntotal,
            self._index.d,
        )

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        """Embed query, normalise, and return top-k chunks by inner product (= cosine similarity).

        Args:
            query: Free-text search query.
            k: Number of results to return.

        Returns:
            List of dicts with keys chunk_text, metadata, and score (cosine similarity, 0–1).
        """
        import faiss
        import numpy as np

        k = min(k, self._index.ntotal)
        if k == 0:
            return []

        vec = self._model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)

        scores, indices = self._index.search(vec, k)

        return [
            {
                "chunk_text": self._chunks[idx]["chunk_text"],
                "metadata": self._chunks[idx]["metadata"],
                "score": round(float(score), 4),
            }
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]
