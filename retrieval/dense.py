"""Dense retriever: loads a ChromaDB collection and searches via embedding similarity."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "chroma")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    """Semantic search over a persistent ChromaDB collection."""

    def __init__(self, collection_name: str = "golgi") -> None:
        """Load an existing ChromaDB collection by name.

        Args:
            collection_name: Name of the collection created by embed_and_store().
        """
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._collection = client.get_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
        )
        logger.info(
            "DenseRetriever: loaded collection '%s' (%d vectors)",
            collection_name,
            self._collection.count(),
        )

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        """Embed query and return top-k most similar chunks.

        Args:
            query: Free-text search query.
            k: Number of results to return.

        Returns:
            List of dicts with keys chunk_text, metadata, and score (cosine similarity).
        """
        k = min(k, self._collection.count())
        if k == 0:
            return []

        results = self._collection.query(query_texts=[query], n_results=k)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        # ChromaDB returns cosine distance (0=identical); convert to similarity
        distances = results["distances"][0]

        return [
            {"chunk_text": doc, "metadata": meta, "score": round(1.0 - dist, 4)}
            for doc, meta, dist in zip(docs, metas, distances)
        ]
