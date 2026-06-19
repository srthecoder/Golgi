"""Sparse retriever: BM25 index over a fixed corpus of text chunks."""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-word characters."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """BM25Okapi keyword search over an in-memory list of text chunks."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        """Build a BM25 index from the provided chunks.

        Args:
            chunks: List of dicts with keys chunk_text and metadata,
                    as produced by chunk_documents().
        """
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        tokenized = [_tokenize(c["chunk_text"]) for c in chunks]
        self._index = BM25Okapi(tokenized)
        logger.info("BM25Retriever: indexed %d chunks", len(chunks))

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        """Return top-k chunks ranked by BM25 score.

        Args:
            query: Free-text search query.
            k: Number of results to return.

        Returns:
            List of dicts with keys chunk_text, metadata, and score (BM25 raw score),
            sorted descending by score.
        """
        qtokens = _tokenize(query)
        scores = self._index.get_scores(qtokens)

        k = min(k, len(self._chunks))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            {
                "chunk_text": self._chunks[i]["chunk_text"],
                "metadata": self._chunks[i]["metadata"],
                "score": round(float(scores[i]), 4),
            }
            for i in top_indices
        ]
