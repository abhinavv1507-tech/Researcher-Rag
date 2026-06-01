"""
utils/reranker.py

Score-gated reranker with MMR diversity pass.
Primary: sentence-transformers CrossEncoder for bge-reranker-v2.5-gemma2-lightweight
Fallback: HuggingFace AutoModelForSequenceClassification + AutoTokenizer (for Gemma 2 arch)
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Any
from utils.logging_config import get_logger

log = get_logger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-v2.5-gemma2-lightweight"
SCORE_THRESHOLD = 0.6
MAX_CHUNKS_PER_SOURCE = 2  # Hard cap: no more than 2 chunks per arxiv_id after score gating


def _load_cross_encoder():
    """Attempt to load via sentence-transformers CrossEncoder."""
    from sentence_transformers import CrossEncoder  # type: ignore
    model = CrossEncoder(RERANKER_MODEL, max_length=512)
    log.info("reranker.loaded", backend="sentence-transformers", model=RERANKER_MODEL)
    return ("cross_encoder", model)


def _load_hf_model():
    """Fallback: load via HuggingFace transformers with accelerate device mapping."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    log.info("reranker.loaded", backend="transformers", model=RERANKER_MODEL)
    return ("hf", model, tokenizer)


def _get_reranker():
    """Return the reranker backend (cached on first call)."""
    if not hasattr(_get_reranker, "_cached"):
        try:
            _get_reranker._cached = _load_cross_encoder()
        except Exception as e:
            log.warning("reranker.fallback", reason=str(e))
            _get_reranker._cached = _load_hf_model()
    return _get_reranker._cached


def _score_pairs(query: str, chunks: list[dict]) -> list[float]:
    """Score (query, chunk_text) pairs using whichever backend loaded."""
    backend = _get_reranker()
    texts = [c["full_contextualized_text"] for c in chunks]
    pairs = [(query, t) for t in texts]

    if backend[0] == "cross_encoder":
        _, model = backend
        scores = model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
    else:
        _, model, tokenizer = backend
        scores: list[float] = []
        for pair in pairs:
            enc = tokenizer(
                pair[0],
                pair[1],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            score = torch.sigmoid(logits[0]).item() if logits.shape[-1] == 1 else float(logits[0][1])
            scores.append(score)
        return scores


def _mmr_diversity(
    scored_chunks: list[dict],
    max_per_source: int = MAX_CHUNKS_PER_SOURCE,
) -> list[dict]:
    """
    Enforce source diversity with a hard cap: keep at most `max_per_source` chunks
    per arxiv_id (default 2). Chunks must be sorted by score descending before calling
    so the highest-scoring chunk from each paper is always preferred.
    """
    source_counts: dict[str, int] = {}
    selected: list[dict] = []
    for chunk in scored_chunks:
        aid = chunk.get("arxiv_id", "unknown")
        if source_counts.get(aid, 0) < max_per_source:
            selected.append(chunk)
            source_counts[aid] = source_counts.get(aid, 0) + 1
    return selected


def rerank(query: str, chunks: list[dict]) -> list[dict]:
    """
    Main reranking pipeline:
    1. Score all chunks against the query.
    2. Drop chunks with score < SCORE_THRESHOLD.
    3. Apply MMR diversity pass (≤ MAX_CHUNKS_PER_SOURCE per arxiv_id).
    4. Return surviving chunks sorted by score descending.

    Args:
        query:  The user's original query string.
        chunks: Pooled list of chunk dicts (from dense + lexical workers).

    Returns:
        Filtered, diversity-capped, score-sorted list of chunk dicts.
    """
    if not chunks:
        return []

    log.info("reranker.scoring", n_chunks=len(chunks))
    scores = _score_pairs(query, chunks)

    # Attach scores and filter
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = score

    gated = [c for c in chunks if c["rerank_score"] >= SCORE_THRESHOLD]
    log.info("reranker.gate", passed=len(gated), dropped=len(chunks) - len(gated), threshold=SCORE_THRESHOLD)

    # Sort by score descending before MMR
    gated.sort(key=lambda c: c["rerank_score"], reverse=True)

    diverse = _mmr_diversity(gated)
    log.info("reranker.mmr", final=len(diverse))
    return diverse
