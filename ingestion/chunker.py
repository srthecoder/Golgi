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


FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "faiss.index")
CHUNKS_META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "chunks_metadata.json")


def embed_and_store(
    chunks: list[dict[str, Any]],
    collection_name: str = "golgi",
):
    """Embed chunks with all-MiniLM-L6-v2, build a FAISS IndexFlatIP index, and persist to disk.

    Vectors are L2-normalised before adding so inner product equals cosine similarity.

    Args:
        chunks: Output of chunk_documents().
        collection_name: Unused; kept for API compatibility.

    Returns:
        The faiss index object.
    """
    import json

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not chunks:
        logger.warning("embed_and_store called with empty chunk list")
        return None

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["chunk_text"] for c in chunks]

    logger.info("Embedding %d chunks with %s...", len(chunks), EMBED_MODEL_NAME)
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")

    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, FAISS_INDEX_PATH)
    logger.info("FAISS index saved to %s (%d vectors, dim=%d)", FAISS_INDEX_PATH, index.ntotal, dim)

    with open(CHUNKS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    logger.info("Chunk metadata saved to %s", CHUNKS_META_PATH)

    return index
