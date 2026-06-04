"""evaluation — RAGAS-style evaluation package."""
from evaluation.ragas_evaluator import RAGASEvaluator
from evaluation.scoring import ConfidenceScorer
from evaluation.metrics import (
    compute_context_precision,
    compute_context_recall,
    compute_citation_coverage,
    extract_citation_ids,
    split_sentences,
)

__all__ = [
    "RAGASEvaluator",
    "ConfidenceScorer",
    "compute_context_precision",
    "compute_context_recall",
    "compute_citation_coverage",
    "extract_citation_ids",
    "split_sentences",
]
