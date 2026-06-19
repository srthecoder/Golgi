"""Chunking and embedding pipeline: splits documents and stores them in ChromaDB."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "chroma")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_documents(
    docs: list[dict[str, Any]],
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """Split document abstracts/descriptions into overlapping text chunks.

    Args:
        docs: List of document dicts with keys id, title, abstract/description, url, source, date.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of dicts with keys chunk_text and metadata (id, title, url, source, date).
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )

    chunks: list[dict[str, Any]] = []
    for doc in docs:
        text = doc.get("abstract") or doc.get("description") or ""
        if not text.strip():
            logger.debug("Skipping doc '%s' — no text content", doc.get("id", "?"))
            continue

        metadata = {
            "id": str(doc.get("id", "")),
            "title": str(doc.get("title", "")),
            "url": str(doc.get("url", "")),
            "source": str(doc.get("source", "")),
            "date": str(doc.get("date", "")),
        }

        parts = splitter.split_text(text)
        for part in parts:
            chunks.append({"chunk_text": part, "metadata": metadata})

    logger.info("chunk_documents: %d docs → %d chunks", len(docs), len(chunks))
    return chunks


def embed_and_store(
    chunks: list[dict[str, Any]],
    collection_name: str = "golgi",
):
    """Embed chunks with all-MiniLM-L6-v2 and persist them in ChromaDB.

    Args:
        chunks: Output of chunk_documents().
        collection_name: Name of the ChromaDB collection.

    Returns:
        The ChromaDB collection object.
    """
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Get or create collection — wipe and recreate so re-runs don't duplicate
    try:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'", collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if not chunks:
        logger.warning("embed_and_store called with empty chunk list")
        return collection

    # ChromaDB requires unique string IDs
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    texts = [c["chunk_text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Add in batches to avoid memory spikes on large corpora
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        logger.debug("Stored batch %d–%d", start, min(end, len(chunks)))

    logger.info("embed_and_store: stored %d chunks in collection '%s'", len(chunks), collection_name)
    return collection
