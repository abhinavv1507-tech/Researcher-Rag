"""
utils/reranker.py

Production reranker: cross-encoder/ms-marco-MiniLM-L-6-v2  (~80 MB, fast CPU)

Pipeline:
  1. Deduplicate by chunk_id
  2. Cap input at MAX_RERANK_INPUT (top-30 from merged top-15 dense + top-15 lexical)
  3. Batch CrossEncoder scoring
  4. Score-gate: keep chunks above SCORE_THRESHOLD
  5. Sort descending by rerank_score
  6. Truncate to RERANK_TOP_K (top-10)
  7. MMR diversity pass: max MAX_CHUNKS_PER_SOURCE per arxiv_id
  8. Final truncation to final_k (default = 8)

Model is loaded once via utils.model_cache (singleton).
"""
from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv

from utils.logging_config import get_logger
from utils.model_cache import get_reranker_model

load_dotenv()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config (Task 9 — increased retrieval bandwidth)
# ---------------------------------------------------------------------------

RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
SCORE_THRESHOLD: float = 0.0         # Raw logits from MiniLM; keep all non-negative
MAX_RERANK_INPUT: int = 30           # Score top-30 combined chunks (was 20)
RERANK_TOP_K: int = 10              # Keep top-10 after cross-encoder scoring
MAX_CHUNKS_PER_SOURCE: int = 3       # MMR diversity: max 3 chunks per arxiv_id (was 2)
BATCH_SIZE: int = 32                 # Inference batch size


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deduplicate(chunks: list[dict]) -> list[dict]:
    """Keep highest-retrieval-score copy of each chunk_id."""
    seen: dict[Any, dict] = {}
    for c in chunks:
        cid = c.get("chunk_id")
        if cid not in seen or c.get("score", 0.0) > seen[cid].get("score", 0.0):
            seen[cid] = c
    return list(seen.values())


def _score_pairs(query: str, chunks: list[dict]) -> list[float]:
    """
    Batch-score (query, chunk_text) pairs using the singleton CrossEncoder.
    Falls back to normalised retrieval scores if the model is unavailable.
    """
    backend, model = get_reranker_model()
    texts = [c.get("full_contextualized_text", "") for c in chunks]

    if backend == "cross_encoder":
        pairs = [(query, t) for t in texts]
        scores: list[float] = []
        for i in range(0, len(pairs), BATCH_SIZE):
            batch_scores = model.predict(pairs[i : i + BATCH_SIZE], show_progress_bar=False)
            scores.extend(float(s) for s in batch_scores)
        return scores
    else:
        # Fallback: normalise retrieval scores to [0, 1]
        raw = [float(c.get("score", 0.5)) for c in chunks]
        lo, hi = min(raw), max(raw)
        span = hi - lo or 1.0
        return [(s - lo) / span for s in raw]


def _mmr_diversity(
    scored_chunks: list[dict],
    max_per_source: int = MAX_CHUNKS_PER_SOURCE,
) -> list[dict]:
    """
    Enforce source diversity: keep at most `max_per_source` chunks per arxiv_id.
    Input must be sorted by rerank_score descending.
    """
    counts: dict[str, int] = {}
    selected: list[dict] = []
    for c in scored_chunks:
        aid = c.get("arxiv_id", "unknown")
        if counts.get(aid, 0) < max_per_source:
            selected.append(c)
            counts[aid] = counts.get(aid, 0) + 1
    return selected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(query: str, chunks: list[dict], final_k: int = 8) -> list[dict]:
    """
    Main reranking pipeline.

    Args:
        query   : Original user query.
        chunks  : Pooled chunks from dense + lexical workers (up to 30).
        final_k : Maximum chunks to return after MMR (default 8, up from 5).

    Returns:
        Score-sorted, diversity-capped list of up to final_k chunk dicts.

    Pipeline:
        dedup → cap to MAX_RERANK_INPUT(30) → cross-encoder score
        → gate → sort → top-RERANK_TOP_K(10) → MMR → top-final_k(8)
    """
    if not chunks:
        return []

    t0 = time.perf_counter()

    # 1. Dedup
    chunks = _deduplicate(chunks)

    # 2. Cap — sort by retrieval score so we keep the best candidates
    chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    chunks = chunks[:MAX_RERANK_INPUT]

    log.info(
        "reranker.scoring",
        n_chunks=len(chunks),
        model=RERANKER_MODEL,
    )

    # 3. Batch score with cross-encoder
    scores = _score_pairs(query, chunks)

    # 4. Attach scores & gate
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = score

    gated = [c for c in chunks if c["rerank_score"] >= SCORE_THRESHOLD]
    log.info(
        "reranker.gate",
        passed=len(gated),
        dropped=len(chunks) - len(gated),
    )

    # 5. Sort by rerank score
    gated.sort(key=lambda c: c["rerank_score"], reverse=True)

    # 6. Keep top-RERANK_TOP_K before MMR (ensures MMR sees most relevant)
    top_k = gated[:RERANK_TOP_K]

    # 7. MMR diversity → final_k
    diverse = _mmr_diversity(top_k)[:final_k]

    elapsed = time.perf_counter() - t0
    log.info("reranker.completed", final=len(diverse), elapsed_s=round(elapsed, 3))
    return diverse
