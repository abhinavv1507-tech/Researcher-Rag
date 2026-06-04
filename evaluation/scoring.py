"""
evaluation/scoring.py

Confidence scorer for the RAG research pipeline.

Confidence formula:
    confidence = 0.35 * faithfulness
               + 0.25 * answer_relevance
               + 0.15 * context_precision
               + 0.15 * context_recall
               + 0.10 * citation_coverage

Bands:
    HIGH   : [0.80, 1.00]
    MEDIUM : [0.50, 0.79]
    LOW    : [0.00, 0.49]

Faithfulness gates:
    < 0.70  → mark LOW regardless of other metrics
    < 0.50  → pipeline should NOT synthesize (return evidence only)
"""
from __future__ import annotations

from typing import Literal

from utils.logging_config import get_logger

log = get_logger(__name__)

# Weighted formula coefficients (must sum to 1.0)
WEIGHTS = {
    "faithfulness": 0.35,
    "answer_relevance": 0.25,
    "context_precision": 0.15,
    "context_recall": 0.15,
    "citation_coverage": 0.10,
}

# Faithfulness thresholds
FAITHFULNESS_LOW_GATE: float = 0.70   # below → mark LOW
FAITHFULNESS_BLOCK_GATE: float = 0.50  # below → do not synthesize

ConfidenceBand = Literal["HIGH", "MEDIUM", "LOW"]


class ConfidenceScorer:
    """
    Compute weighted confidence score from RAGAS metrics.

    Usage:
        scorer = ConfidenceScorer()
        result = scorer.score(metrics_dict)
        # result = {"confidence": 0.82, "band": "HIGH", "should_synthesize": True}
    """

    def score(self, metrics: dict[str, float]) -> dict:
        """
        Compute confidence from RAGAS metrics.

        Args:
            metrics: dict with keys matching WEIGHTS above

        Returns:
            {
                "confidence": float,          # 0.0 – 1.0
                "band": "HIGH"|"MEDIUM"|"LOW",
                "should_synthesize": bool,    # False if faithfulness < 0.50
                "faithfulness_warning": bool, # True if faithfulness < 0.70
            }
        """
        faithfulness = metrics.get("faithfulness", 0.0)

        # Compute weighted score
        confidence = sum(
            WEIGHTS[k] * metrics.get(k, 0.0)
            for k in WEIGHTS
        )
        confidence = round(min(1.0, max(0.0, confidence)), 4)

        # Determine band
        band: ConfidenceBand
        if faithfulness < FAITHFULNESS_LOW_GATE:
            # Faithfulness gate overrides band calculation
            band = "LOW"
        elif confidence >= 0.80:
            band = "HIGH"
        elif confidence >= 0.50:
            band = "MEDIUM"
        else:
            band = "LOW"

        should_synthesize = faithfulness >= FAITHFULNESS_BLOCK_GATE
        faithfulness_warning = faithfulness < FAITHFULNESS_LOW_GATE

        log.info(
            "confidence_scorer.result",
            confidence=confidence,
            band=band,
            faithfulness=round(faithfulness, 3),
            should_synthesize=should_synthesize,
            faithfulness_warning=faithfulness_warning,
        )

        return {
            "confidence": confidence,
            "band": band,
            "should_synthesize": should_synthesize,
            "faithfulness_warning": faithfulness_warning,
        }

    @staticmethod
    def describe_band(band: ConfidenceBand) -> str:
        """Human-readable description of confidence band."""
        return {
            "HIGH": "High Confidence (0.80 – 1.00): Answer strongly grounded in evidence.",
            "MEDIUM": "Medium Confidence (0.50 – 0.79): Answer mostly grounded; some gaps possible.",
            "LOW": "Low Confidence (0.00 – 0.49): Answer may not be fully supported by evidence.",
        }.get(band, "Unknown")
