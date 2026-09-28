"""Sentence-embedding model, loaded once and reused."""

from __future__ import annotations

import logging
import threading

from app.config import settings

logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def resolve_device() -> str:
    import torch

    if settings.device != "auto":
        return settings.device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embedder():
    """Thread-safe lazy load of the SentenceTransformer model."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                device = resolve_device()
                logger.info(
                    "Loading embedding model %s on %s", settings.embedding_model, device
                )
                _model = SentenceTransformer(settings.embedding_model, device=device)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    model = get_embedder()
    prefixed = f"{settings.query_prefix}{text}" if settings.query_prefix else text
    vector = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vector.tolist()


def embedding_dimension() -> int:
    return int(get_embedder().get_sentence_embedding_dimension())
