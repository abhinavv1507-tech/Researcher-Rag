"""
contextual_retrieval/service.py

ContextualRetrievalService — orchestrates contextual chunk enrichment
for the full ingest pipeline.

Usage:
    from contextual_retrieval.service import ContextualRetrievalService
    service = ContextualRetrievalService(use_llm=True)
    enriched_chunks = service.enrich(chunks, paper_metadata)
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

from contextual_retrieval.contextualizer import contextualize_chunks
from utils.logging_config import get_logger

log = get_logger(__name__)

# How many Groq calls to make per second (rate-limit safety)
# llama-3.1-8b-instant: 30k tokens/min on free tier
# Each context call ≈ 300 tokens → safe at ~1/sec per key
INTER_PAPER_DELAY_S: float = float(os.getenv("CONTEXT_INTER_PAPER_DELAY_S", "0.5"))


class ContextualRetrievalService:
    """
    Enrich a flat list of chunks with Anthropic-style contextual retrieval context.

    The service:
    1. Groups chunks by paper (arxiv_id)
    2. Calls contextualize_chunks() for each paper
    3. Returns a flat list of enriched chunks in original order
    """

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm
        log.info("contextual_retrieval_service.init", use_llm=use_llm)

    def enrich(
        self,
        chunks: list[dict[str, Any]],
        paper_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Enrich all chunks with contextual retrieval context.

        Args:
            chunks: Flat list of chunk dicts from ingest pipeline
            paper_metadata: Optional {arxiv_id: {title, abstract, ...}} mapping.
                            If None, metadata is extracted from chunk fields.

        Returns:
            Flat list of enriched chunk dicts (same length, same order)
        """
        t_start = time.perf_counter()
        log.info("contextual_retrieval_service.enrich_start", n_chunks=len(chunks))

        # Group chunks by paper
        paper_chunks: dict[str, list[dict]] = defaultdict(list)
        chunk_order: dict[int, int] = {}  # chunk_id → position in output

        for pos, chunk in enumerate(chunks):
            arxiv_id = chunk.get("arxiv_id", chunk.get("paper_id", "unknown"))
            paper_chunks[arxiv_id].append(chunk)
            chunk_order[chunk.get("chunk_id", pos)] = pos

        # Sort each paper's chunks by chunk_index
        for aid in paper_chunks:
            paper_chunks[aid].sort(key=lambda c: c.get("chunk_index", 0))

        enriched_map: dict[int, dict] = {}
        papers_done = 0

        for arxiv_id, p_chunks in paper_chunks.items():
            # Build paper metadata dict
            if paper_metadata and arxiv_id in paper_metadata:
                paper_info = paper_metadata[arxiv_id]
            else:
                # Extract from first chunk
                first = p_chunks[0]
                paper_info = {
                    "arxiv_id": arxiv_id,
                    "title": first.get("title", ""),
                    "abstract": first.get("context_summary", ""),
                }

            try:
                enriched_paper_chunks = contextualize_chunks(
                    paper=paper_info,
                    chunks=p_chunks,
                    use_llm=self.use_llm,
                )
            except Exception as exc:
                log.error(
                    "contextual_retrieval_service.paper_failed",
                    arxiv_id=arxiv_id,
                    error=str(exc),
                )
                enriched_paper_chunks = p_chunks  # use originals as fallback

            for chunk in enriched_paper_chunks:
                cid = chunk.get("chunk_id")
                enriched_map[cid] = chunk

            papers_done += 1
            if self.use_llm and papers_done < len(paper_chunks):
                time.sleep(INTER_PAPER_DELAY_S)

        # Reconstruct in original order
        result: list[dict] = []
        for chunk in chunks:
            cid = chunk.get("chunk_id")
            result.append(enriched_map.get(cid, chunk))

        elapsed = time.perf_counter() - t_start
        log.info(
            "contextual_retrieval_service.enrich_done",
            n_chunks=len(result),
            papers=papers_done,
            elapsed_s=round(elapsed, 2),
        )
        return result
