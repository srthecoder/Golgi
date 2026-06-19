"""Cross-encoder reranker using ms-marco-MiniLM-L-6-v2."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks retrieval candidates with a cross-encoder relevance model."""

    def __init__(self) -> None:
        """Load the cross-encoder model."""
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(MODEL_NAME)
        logger.info("CrossEncoderReranker: loaded %s", MODEL_NAME)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Score all query-candidate pairs and return top_k by cross-encoder score.

        Args:
            query: The search query.
            candidates: List of dicts with at least chunk_text and metadata keys.
            top_k: Number of top results to return.

        Returns:
            List of dicts with chunk_text, metadata, and score (cross-encoder logit),
            sorted descending by score, length <= top_k.
        """
        if not candidates:
            return []

        pairs = [(query, c["chunk_text"]) for c in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            {
                "chunk_text": c["chunk_text"],
                "metadata": c["metadata"],
                "score": round(float(s), 4),
            }
            for s, c in ranked[:top_k]
        ]
