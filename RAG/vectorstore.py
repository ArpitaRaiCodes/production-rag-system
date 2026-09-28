"""Persistent vector store backed by ChromaDB.

Everything the UI needs to manage the knowledge base lives here: adding a
document, listing what is stored, deleting one document, and wiping the whole
collection.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.embeddings import embed_documents, embed_query

logger = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_lock = threading.Lock()


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                logger.info("Opening Chroma store at %s", settings.chroma_dir)
                _client = chromadb.PersistentClient(
                    path=str(settings.chroma_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
    return _client


def get_collection():
    """Cosine space, and no Chroma-side embedding function: we always pass our
    own vectors so the embedding model stays under our control."""
    return get_client().get_or_create_collection(
        name=settings.collection_name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
def add_document(
    doc_id: str,
    filename: str,
    chunks: list[dict[str, Any]],
) -> int:
    """Embed and store every chunk of one document. Returns the chunk count."""
    collection = get_collection()
    texts = [c["text"] for c in chunks]
    vectors = embed_documents(texts)
    uploaded_at = datetime.now(timezone.utc).isoformat()

    ids = [f"{doc_id}:{c['index']}" for c in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": int(c["index"]),
            "page": int(c["page"]) if c.get("page") else 0,
            "uploaded_at": uploaded_at,
            "n_chars": len(c["text"]),
        }
        for c in chunks
    ]

    # Chroma is happiest with moderate batches.
    batch = 128
    for start in range(0, len(ids), batch):
        stop = start + batch
        collection.add(
            ids=ids[start:stop],
            documents=texts[start:stop],
            embeddings=vectors[start:stop],
            metadatas=metadatas[start:stop],
        )
    logger.info("Stored %d chunks for %s (%s)", len(ids), filename, doc_id)
    return len(ids)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Return the top-k chunks for a query, best match first."""
    top_k = top_k or settings.top_k
    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict[str, Any]] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for rank, (text, meta, distance) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        # Chroma cosine distance -> similarity
        score = round(1.0 - float(distance), 4)
        hits.append(
            {
                "rank": rank,
                "score": score,
                "text": text,
                "doc_id": meta.get("doc_id"),
                "filename": meta.get("filename"),
                "page": meta.get("page") or None,
                "chunk_index": meta.get("chunk_index"),
            }
        )
    return hits


def list_documents() -> list[dict[str, Any]]:
    """One row per uploaded file, newest first."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    stored = collection.get(include=["metadatas"])
    grouped: dict[str, dict[str, Any]] = {}
    for meta in stored.get("metadatas", []):
        doc_id = meta.get("doc_id")
        if not doc_id:
            continue
        entry = grouped.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "filename": meta.get("filename", "unknown"),
                "uploaded_at": meta.get("uploaded_at"),
                "chunks": 0,
                "characters": 0,
            },
        )
        entry["chunks"] += 1
        entry["characters"] += int(meta.get("n_chars") or 0)

    return sorted(
        grouped.values(), key=lambda d: d.get("uploaded_at") or "", reverse=True
    )


def document_exists(doc_id: str) -> bool:
    found = get_collection().get(where={"doc_id": doc_id}, limit=1, include=[])
    return bool(found.get("ids"))


def stats() -> dict[str, Any]:
    collection = get_collection()
    documents = list_documents()
    return {
        "collection": settings.collection_name,
        "documents": len(documents),
        "chunks": collection.count(),
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "storage_path": str(settings.chroma_dir),
    }


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------
def delete_document(doc_id: str) -> int:
    """Remove every chunk belonging to one document. Returns chunks removed."""
    collection = get_collection()
    existing = collection.get(where={"doc_id": doc_id}, include=[])
    ids = existing.get("ids", [])
    if not ids:
        return 0
    collection.delete(ids=ids)
    logger.info("Deleted %d chunks for doc_id=%s", len(ids), doc_id)
    return len(ids)


def reset_collection() -> int:
    """Drop the whole collection and recreate it empty."""
    client = get_client()
    collection = get_collection()
    removed = collection.count()
    client.delete_collection(settings.collection_name)
    get_collection()  # recreate immediately so the app stays usable
    logger.info("Reset collection, removed %d chunks", removed)
    return removed
