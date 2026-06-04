"""
utils/model_cache.py

Singleton/global model loading for embedding model and reranker.
Both models are loaded once per process, cached, and reused everywhere.
Load time and model metadata are logged on first load.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from dotenv import load_dotenv

from utils.logging_config import get_logger

load_dotenv()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config — read from environment with sensible defaults
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME: str = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RERANKER_MODEL_NAME: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ---------------------------------------------------------------------------
# Embedding model singleton
# ---------------------------------------------------------------------------

_embedding_model: Optional[Any] = None


def get_embedding_model():
    """Return the cached SentenceTransformer embedding model (loads once)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        log.info("model_cache.embedding.loading", model=EMBED_MODEL_NAME)
        t0 = time.perf_counter()
        _embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
        elapsed = time.perf_counter() - t0
        log.info(
            "model_cache.embedding.loaded",
            model=EMBED_MODEL_NAME,
            load_time_s=round(elapsed, 2),
        )
    return _embedding_model


# ---------------------------------------------------------------------------
# Reranker model singleton
# ---------------------------------------------------------------------------

_reranker_model: Optional[Any] = None
_reranker_backend: str = "none"


def get_reranker_model() -> tuple[str, Any]:
    """
    Return (backend_name, model) for the cached reranker (loads once).
    backend_name is 'cross_encoder' or 'none' (fallback).
    """
    global _reranker_model, _reranker_backend

    if _reranker_model is None and _reranker_backend == "none":
        from sentence_transformers import CrossEncoder  # type: ignore

        log.info("model_cache.reranker.loading", model=RERANKER_MODEL_NAME)
        t0 = time.perf_counter()
        try:
            _reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)
            _reranker_backend = "cross_encoder"
            elapsed = time.perf_counter() - t0
            log.info(
                "model_cache.reranker.loaded",
                model=RERANKER_MODEL_NAME,
                backend="cross_encoder",
                load_time_s=round(elapsed, 2),
            )
        except Exception as exc:
            log.warning(
                "model_cache.reranker.load_failed",
                model=RERANKER_MODEL_NAME,
                error=str(exc),
                fallback="using hybrid scores only",
            )
            _reranker_backend = "none"

    return (_reranker_backend, _reranker_model)
