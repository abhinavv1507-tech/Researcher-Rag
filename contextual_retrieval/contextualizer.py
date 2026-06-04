"""
contextual_retrieval/contextualizer.py

Core Anthropic-style chunk contextualizer.

For every chunk in a document, generates 2-4 sentences of retrieval context
using: document title + abstract summary + section title + neighboring chunks.

Uses Groq with key rotation and graceful fallback to abstract-based context
when LLM is unavailable.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
from langchain_groq import ChatGroq  # type: ignore

from contextual_retrieval.prompts import (
    CONTEXTUALIZER_SYSTEM_PROMPT,
    CONTEXTUALIZER_USER_TEMPLATE,
    FALLBACK_CONTEXT_TEMPLATE,
)
from utils.groq_key_manager import get_key_manager
from utils.logging_config import get_logger

log = get_logger(__name__)

# Model for context generation — use fast model to save quota
CONTEXT_MODEL: str = os.getenv("GROQ_CONTEXT_MODEL", "llama-3.1-8b-instant")
MAX_CHUNK_TEXT_LEN: int = 800   # truncate chunk for context prompt
MAX_SUMMARY_LEN: int = 300      # truncate abstract for prompt
MAX_RETRIES: int = 3


def _extract_section_title(
    chunks: list[dict],
    chunk_index: int,
    window: int = 1,
) -> str:
    """
    Heuristic section title extraction: look at neighboring chunks
    for lines that look like section headers (short, ALL CAPS or Title Case).
    """
    candidates: list[str] = []
    for i in range(max(0, chunk_index - window), min(len(chunks), chunk_index + window + 1)):
        text = chunks[i].get("chunk_text", "")
        for line in text.split("\n")[:5]:
            line = line.strip()
            if 5 < len(line) < 80 and (line.isupper() or line.istitle()):
                candidates.append(line)
    return candidates[0] if candidates else "Main Content"


def _groq_generate_context(
    title: str,
    summary: str,
    section_title: str,
    chunk_text: str,
    max_retries: int = MAX_RETRIES,
) -> Optional[str]:
    """
    Call Groq to generate 2-4 sentences of retrieval context.
    Returns None if all retries fail.
    """
    user_content = CONTEXTUALIZER_USER_TEMPLATE.format(
        title=title,
        summary=summary[:MAX_SUMMARY_LEN],
        section_title=section_title,
        chunk_text=chunk_text[:MAX_CHUNK_TEXT_LEN],
    )
    messages = [
        SystemMessage(content=CONTEXTUALIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    for attempt in range(max_retries):
        key = get_key_manager().next_key()
        try:
            llm = ChatGroq(
                model=CONTEXT_MODEL,
                temperature=0.0,
                max_tokens=200,
                api_key=key,
            )
            response = llm.invoke(messages)
            context = response.content.strip()
            if context:
                return context
        except Exception as exc:
            mgr = get_key_manager()
            msg = str(exc).lower()
            if "429" in msg or "rate limit" in msg:
                mgr.mark_rate_limited(key)
                time.sleep(2 ** attempt)  # brief backoff
            elif "401" in msg or "invalid_api_key" in msg:
                mgr.mark_failed(key)
            log.warning(
                "contextualizer.retry",
                attempt=attempt + 1,
                max=max_retries,
                error=str(exc)[:100],
            )
    return None


def _fallback_context(title: str, summary: str) -> str:
    """Generate simple abstract-based context when LLM is unavailable."""
    summary_short = summary[:150].rstrip()
    return FALLBACK_CONTEXT_TEMPLATE.format(
        title=title, summary_short=summary_short
    )


def contextualize_chunks(
    paper: dict[str, Any],
    chunks: list[dict[str, Any]],
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate Anthropic-style contextual retrieval context for all chunks
    in a single paper.

    Args:
        paper: Paper metadata dict with keys: arxiv_id, title, abstract
        chunks: All chunks belonging to this paper (ordered by chunk_index)
        use_llm: If False, use fallback context only (no Groq calls)

    Returns:
        List of chunk dicts with added/updated fields:
            - context: str (LLM-generated or fallback)
            - embedding_text: str (context + chunk_text)
            - full_contextualized_text: str (same as embedding_text)
            - context_source: 'llm' | 'fallback'
    """
    title = paper.get("title", "Unknown Paper")
    summary = paper.get("abstract", paper.get("context_summary", ""))
    arxiv_id = paper.get("arxiv_id", "")

    enriched: list[dict[str, Any]] = []
    llm_success = 0
    fallback_count = 0

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("chunk_text", chunk.get("full_contextualized_text", ""))
        section_title = _extract_section_title(chunks, i)

        context: str
        source: str

        if use_llm:
            generated = _groq_generate_context(
                title=title,
                summary=summary,
                section_title=section_title,
                chunk_text=chunk_text,
            )
            if generated:
                context = generated
                source = "llm"
                llm_success += 1
            else:
                context = _fallback_context(title, summary)
                source = "fallback"
                fallback_count += 1
        else:
            context = _fallback_context(title, summary)
            source = "fallback"
            fallback_count += 1

        embedding_text = f"{context}\n\n{chunk_text}"

        updated = dict(chunk)
        updated["context"] = context
        updated["embedding_text"] = embedding_text
        updated["full_contextualized_text"] = embedding_text
        updated["context_source"] = source
        enriched.append(updated)

    log.info(
        "contextualizer.paper_done",
        arxiv_id=arxiv_id,
        total=len(chunks),
        llm_generated=llm_success,
        fallback=fallback_count,
    )
    return enriched
