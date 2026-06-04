"""
evaluation/ragas_evaluator.py

RAGAS-inspired evaluation layer for the RAG research pipeline.

Computes 5 metrics:
  1. Faithfulness       — LLM-based: are claims supported by retrieved evidence?
  2. Answer Relevance   — LLM-based: does the answer address the query?
  3. Context Precision  — Heuristic: fraction of chunks relevant to query
  4. Context Recall     — Heuristic: fraction of answer sentences with chunk support
  5. Citation Coverage  — Structural: fraction of cited sentences with valid citations

No RAGAS package dependency — implemented directly with Groq + heuristics.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
from langchain_groq import ChatGroq  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from evaluation.metrics import (
    compute_citation_coverage,
    compute_context_precision,
    compute_context_recall,
    split_sentences,
)
from utils.groq_key_manager import get_key_manager
from utils.logging_config import get_logger

log = get_logger(__name__)

# Fast model for evaluation — minimises latency + key pressure
EVAL_MODEL: str = os.getenv("GROQ_EVAL_MODEL", "llama-3.1-8b-instant")
MAX_RETRIES: int = 3
MAX_CLAIMS_PER_EVAL: int = 10  # cap claims sent to LLM per faithfulness call


# ---------------------------------------------------------------------------
# Pydantic schemas for LLM evaluation outputs
# ---------------------------------------------------------------------------

class FaithfulnessResult(BaseModel):
    supported_count: int = Field(
        description="Number of claims supported by the provided evidence"
    )
    total_count: int = Field(description="Total number of claims evaluated")
    reasoning: str = Field(description="One-sentence justification")


class RelevanceResult(BaseModel):
    score: float = Field(
        ge=0.0, le=1.0,
        description="Score from 0 to 1: how well the answer addresses the query"
    )
    reasoning: str = Field(description="One-sentence justification")


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _groq_llm(model: str = EVAL_MODEL) -> ChatGroq:
    key = get_key_manager().next_key()
    return ChatGroq(model=model, temperature=0.0, api_key=key)


def _invoke_with_retry(
    schema_class,
    messages: list,
    max_retries: int = MAX_RETRIES,
) -> Optional[Any]:
    """Invoke structured Groq call with key rotation. Returns None on total failure."""
    for attempt in range(max_retries):
        key = get_key_manager().next_key()
        try:
            llm = ChatGroq(
                model=EVAL_MODEL,
                temperature=0.0,
                api_key=key,
            ).with_structured_output(schema_class)
            return llm.invoke(messages)
        except Exception as exc:
            mgr = get_key_manager()
            msg = str(exc).lower()
            if "429" in msg or "rate limit" in msg:
                mgr.mark_rate_limited(key)
                time.sleep(2 ** attempt)
            elif "401" in msg:
                mgr.mark_failed(key)
            log.warning(
                "ragas_evaluator.retry",
                attempt=attempt + 1,
                error=str(exc)[:100],
            )
    return None


# ---------------------------------------------------------------------------
# Faithfulness Evaluator
# ---------------------------------------------------------------------------

def _evaluate_faithfulness(
    claims: list[str],
    evidence_texts: list[str],
) -> float:
    """
    LLM-based faithfulness: check each claim against retrieved evidence.
    Returns fraction of claims that are supported [0, 1].
    """
    if not claims or not evidence_texts:
        return 0.0

    # Cap to avoid huge prompts
    eval_claims = claims[:MAX_CLAIMS_PER_EVAL]
    evidence_context = "\n\n".join(evidence_texts[:8])  # top 8 chunks

    claims_formatted = "\n".join(f"{i+1}. {c}" for i, c in enumerate(eval_claims))

    messages = [
        SystemMessage(
            content=(
                "You are evaluating faithfulness of claims against retrieved evidence.\n"
                "For each claim, determine if it is EXPLICITLY supported by the provided evidence.\n"
                "Do NOT accept claims supported by general knowledge — only by the evidence text.\n"
                "Return JSON with:\n"
                "  supported_count: number of claims fully supported\n"
                "  total_count: total number of claims\n"
                "  reasoning: one-sentence summary"
            )
        ),
        HumanMessage(
            content=(
                f"Claims to evaluate:\n{claims_formatted}\n\n"
                f"Retrieved evidence:\n{evidence_context[:3000]}"
            )
        ),
    ]

    result = _invoke_with_retry(FaithfulnessResult, messages)
    if result is None:
        log.warning("faithfulness.eval_failed — defaulting to 0.5")
        return 0.5  # uncertain

    if result.total_count == 0:
        return 1.0

    score = min(1.0, result.supported_count / result.total_count)
    log.info(
        "faithfulness.computed",
        supported=result.supported_count,
        total=result.total_count,
        score=round(score, 4),
    )
    return round(score, 4)


# ---------------------------------------------------------------------------
# Answer Relevance Evaluator
# ---------------------------------------------------------------------------

def _evaluate_answer_relevance(query: str, answer: str) -> float:
    """
    LLM-based answer relevance: does the answer address the query?
    Returns score in [0, 1].
    """
    if not answer.strip():
        return 0.0

    messages = [
        SystemMessage(
            content=(
                "You are evaluating answer relevance to a research query.\n"
                "Score from 0.0 to 1.0:\n"
                "  1.0 = answer fully and directly addresses the query\n"
                "  0.7 = answer mostly relevant with minor gaps\n"
                "  0.5 = answer partially relevant\n"
                "  0.3 = answer tangentially related\n"
                "  0.0 = answer does not address the query at all\n"
                "Return JSON with 'score' (float) and 'reasoning' (string)."
            )
        ),
        HumanMessage(
            content=(
                f"Research query: {query}\n\n"
                f"Answer (first 1000 chars): {answer[:1000]}"
            )
        ),
    ]

    result = _invoke_with_retry(RelevanceResult, messages)
    if result is None:
        log.warning("answer_relevance.eval_failed — defaulting to 0.5")
        return 0.5

    log.info("answer_relevance.computed", score=result.score)
    return round(result.score, 4)


# ---------------------------------------------------------------------------
# Extract factual claims from answer text
# ---------------------------------------------------------------------------

def _extract_claims(answer: str) -> list[str]:
    """
    Extract individual factual claims from answer sentences.
    Strips citation markers before returning.
    """
    sentences = split_sentences(answer)
    # Remove citation markers for cleaner claim evaluation
    claims = [re.sub(r"\[arXiv:[\d.]+\]", "", s).strip() for s in sentences]
    return [c for c in claims if len(c) > 20]  # filter trivial


# ---------------------------------------------------------------------------
# Main Evaluator Class
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """
    RAGAS-inspired evaluation layer.

    Usage:
        evaluator = RAGASEvaluator()
        metrics = evaluator.evaluate(
            query=query,
            answer=answer,
            chunks=reranked_chunks,
            valid_citations=valid_citations,
        )
    """

    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: list[dict[str, Any]],
        valid_citations: list[str],
        evidence: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, float]:
        """
        Run all 5 RAGAS-style metrics.

        Returns dict:
            {
                "faithfulness": float,
                "answer_relevance": float,
                "context_precision": float,
                "context_recall": float,
                "citation_coverage": float,
            }
        """
        t0 = time.perf_counter()
        log.info("ragas_evaluator.start", query=query[:80])

        # --- Heuristic metrics (fast, no LLM) ---
        context_precision = compute_context_precision(query, chunks)
        context_recall = compute_context_recall(answer, chunks)
        citation_coverage = compute_citation_coverage(answer, valid_citations)

        # --- LLM metrics ---
        # Build evidence texts from chunks
        evidence_texts = [
            c.get("full_contextualized_text", "")
            for c in chunks
            if c.get("full_contextualized_text")
        ]

        # Also include extracted claims from evidence extraction if available
        if evidence:
            for ev in evidence:
                claims_text = "; ".join(ev.get("explicit_claims", []))
                if claims_text:
                    evidence_texts.append(claims_text)

        claims = _extract_claims(answer)
        faithfulness = _evaluate_faithfulness(claims, evidence_texts)
        answer_relevance = _evaluate_answer_relevance(query, answer)

        elapsed = time.perf_counter() - t0
        metrics = {
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "citation_coverage": citation_coverage,
        }

        log.info(
            "ragas_evaluator.done",
            elapsed_s=round(elapsed, 3),
            **{k: round(v, 3) for k, v in metrics.items()},
        )
        return metrics
