"""
evaluation/metrics.py

Pure-function metric computations for RAGAS-style evaluation.
No LLM dependencies — these are heuristic / structural metrics
computed from the pipeline state directly.
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Sentence tokenizer (no nltk dependency)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter using regex.
    Splits on '. ', '! ', '? ' while handling arXiv citation markers.
    """
    # Protect citation markers from splitting
    protected = re.sub(r"\[arXiv:([\d.]+)\]", r"[CITE-\1]", text)
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", protected)
    # Restore citation markers
    restored = [re.sub(r"\[CITE-([\d.]+)\]", r"[arXiv:\1]", s) for s in sentences]
    return [s.strip() for s in restored if s.strip() and len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# Metric: Context Precision
# ---------------------------------------------------------------------------

def compute_context_precision(
    query: str,
    chunks: list[dict[str, Any]],
) -> float:
    """
    Context Precision: what fraction of retrieved chunks are query-relevant?

    A chunk is considered relevant if it shares ≥2 non-trivial tokens
    with the query (after stopword filtering).

    Returns float in [0, 1].
    """
    if not chunks:
        return 0.0

    STOPWORDS = {
        "the", "a", "an", "is", "in", "of", "to", "and", "for",
        "on", "at", "by", "with", "as", "it", "its", "are", "was",
        "be", "do", "have", "from", "this", "that", "how", "what",
        "which", "when", "where", "who", "why", "can", "use",
    }

    query_tokens = {
        t for t in re.findall(r"\b[a-z]{3,}\b", query.lower())
        if t not in STOPWORDS
    }

    if not query_tokens:
        return 1.0  # can't evaluate — give benefit of doubt

    relevant = 0
    for chunk in chunks:
        text = chunk.get("full_contextualized_text", "").lower()
        chunk_tokens = set(re.findall(r"\b[a-z]{3,}\b", text))
        overlap = query_tokens & chunk_tokens
        if len(overlap) >= 2:
            relevant += 1

    return round(relevant / len(chunks), 4)


# ---------------------------------------------------------------------------
# Metric: Context Recall
# ---------------------------------------------------------------------------

def compute_context_recall(
    answer: str,
    chunks: list[dict[str, Any]],
) -> float:
    """
    Context Recall: what fraction of answer sentences have at least
    one supporting chunk?

    A chunk 'supports' a sentence if they share ≥3 non-trivial tokens.

    Returns float in [0, 1].
    """
    if not answer.strip() or not chunks:
        return 0.0

    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    STOPWORDS = {
        "the", "a", "an", "is", "in", "of", "to", "and", "for",
        "on", "at", "by", "with", "as", "it", "its", "are", "was",
        "be", "do", "have", "from", "this", "that",
    }

    chunk_texts = [
        chunk.get("full_contextualized_text", "").lower()
        for chunk in chunks
    ]

    supported = 0
    for sentence in sentences:
        sent_tokens = {
            t for t in re.findall(r"\b[a-z]{4,}\b", sentence.lower())
            if t not in STOPWORDS
        }
        if not sent_tokens:
            supported += 1  # trivial sentence, count as supported
            continue
        for ct in chunk_texts:
            chunk_tokens = set(re.findall(r"\b[a-z]{4,}\b", ct))
            if len(sent_tokens & chunk_tokens) >= 3:
                supported += 1
                break

    return round(supported / len(sentences), 4)


# ---------------------------------------------------------------------------
# Metric: Citation Coverage
# ---------------------------------------------------------------------------

CITATION_PATTERN = re.compile(r"\[arXiv:([\d.]+)\]")


def compute_citation_coverage(
    answer: str,
    valid_citations: list[str],
) -> float:
    """
    Citation Coverage: percentage of answer sentences that contain
    a citation marker AND that citation is in valid_citations.

    Returns float in [0, 1].
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    valid_set = set(valid_citations)
    covered = 0
    total_with_citation_opportunity = 0

    for sentence in sentences:
        found = CITATION_PATTERN.findall(sentence)
        if found:
            total_with_citation_opportunity += 1
            # Sentence has a citation — check if any citation is valid
            if any(cid in valid_set for cid in found):
                covered += 1

    if total_with_citation_opportunity == 0:
        # No citation markers in answer at all
        return 0.0

    return round(covered / total_with_citation_opportunity, 4)


# ---------------------------------------------------------------------------
# Helper: Extract citation IDs from answer text
# ---------------------------------------------------------------------------

def extract_citation_ids(text: str) -> list[str]:
    """Extract all [arXiv:XXXX.XXXXX] citation IDs from text."""
    return CITATION_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Helper: Split answer into sentences (re-exported)
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Public wrapper around _split_sentences."""
    return _split_sentences(text)
