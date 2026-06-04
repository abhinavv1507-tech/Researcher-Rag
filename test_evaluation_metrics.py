"""
test_evaluation_metrics.py — Unit tests for the RAGAS-style metrics
and confidence scorer. No external deps, no LLM calls.
"""
import sys
sys.path.insert(0, ".")

from evaluation.metrics import (
    compute_context_precision,
    compute_context_recall,
    compute_citation_coverage,
    split_sentences,
    extract_citation_ids,
)
from evaluation.scoring import ConfidenceScorer

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

def check(label, condition, got=None):
    if condition:
        print(f"  {PASS} {label}")
    else:
        msg = f"  {FAIL} {label}" + (f" — got: {got}" if got is not None else "")
        print(msg)
        errors.append(label)

print("=== Unit Tests: evaluation.metrics ===\n")

# T1: split_sentences
text = "LLM agents use chain-of-thought reasoning [arXiv:2301.00234]. They also use tool calling [arXiv:2304.09797]. This is a key insight."
sentences = split_sentences(text)
print(f"[T1] split_sentences: {len(sentences)} sentences")
check("split_sentences returns 3 sentences", len(sentences) == 3, sentences)

# T2: extract_citation_ids
ids = extract_citation_ids(text)
print(f"\n[T2] extract_citation_ids: {ids}")
check("extracts 2301.00234", "2301.00234" in ids)
check("extracts 2304.09797", "2304.09797" in ids)

# T3: context_precision — relevant chunks
chunks = [
    {"full_contextualized_text": "LLM agents use tool selection methods for planning"},
    {"full_contextualized_text": "Large language models select tools using retrieval"},
    {"full_contextualized_text": "Totally unrelated content about biology and frogs"},
]
query = "How do LLM agents select tools?"
prec = compute_context_precision(query, chunks)
print(f"\n[T3] context_precision: {prec}")
check("context_precision > 0.5 for relevant chunks", prec > 0.5, prec)

# T4: context_precision — empty chunks
prec_empty = compute_context_precision(query, [])
print(f"\n[T4] context_precision empty chunks: {prec_empty}")
check("context_precision=0.0 for empty chunks", prec_empty == 0.0, prec_empty)

# T5: context_recall
answer = "LLM agents use tool selection through retrieval augmented generation. They also apply chain of thought reasoning methods."
recall = compute_context_recall(answer, chunks[:2])
print(f"\n[T5] context_recall: {recall}")
check("context_recall >= 0.0", recall >= 0.0, recall)
check("context_recall <= 1.0", recall <= 1.0, recall)

# T6: citation_coverage — all valid
answer2 = "Agents use ReAct [arXiv:2210.03629]. They also apply chain-of-thought [arXiv:2201.11903]."
valid_all = ["2210.03629", "2201.11903"]
cov_all = compute_citation_coverage(answer2, valid_all)
print(f"\n[T6] citation_coverage (all valid): {cov_all}")
check("citation_coverage=1.0 when all citations valid", cov_all == 1.0, cov_all)

# T7: citation_coverage — partial valid
valid_partial = ["2210.03629"]
cov_partial = compute_citation_coverage(answer2, valid_partial)
print(f"\n[T7] citation_coverage (partial valid): {cov_partial}")
check("citation_coverage=0.5 when half valid", cov_partial == 0.5, cov_partial)

# T8: citation_coverage — no citations in answer
answer_no_cite = "LLM agents use various retrieval methods."
cov_none = compute_citation_coverage(answer_no_cite, valid_all)
print(f"\n[T8] citation_coverage (no citations in answer): {cov_none}")
check("citation_coverage=0.0 when no [arXiv:...] in answer", cov_none == 0.0, cov_none)

# T9: ConfidenceScorer HIGH
print("\n=== Unit Tests: evaluation.scoring ===\n")
scorer = ConfidenceScorer()
r_high = scorer.score({
    "faithfulness": 0.92,
    "answer_relevance": 0.88,
    "context_precision": 0.85,
    "context_recall": 0.87,
    "citation_coverage": 1.00,
})
print(f"[T9] HIGH confidence: confidence={r_high['confidence']}, band={r_high['band']}")
check("band=HIGH for strong metrics", r_high["band"] == "HIGH", r_high["band"])
check("should_synthesize=True for faithfulness=0.92", r_high["should_synthesize"] == True)
check("confidence >= 0.80", r_high["confidence"] >= 0.80, r_high["confidence"])

# T10: ConfidenceScorer faithfulness block gate
r_block = scorer.score({
    "faithfulness": 0.45,
    "answer_relevance": 0.90,
    "context_precision": 0.88,
    "context_recall": 0.90,
    "citation_coverage": 1.00,
})
print(f"\n[T10] Faithfulness block gate: band={r_block['band']}, should_synthesize={r_block['should_synthesize']}")
check("band=LOW when faithfulness < 0.50", r_block["band"] == "LOW", r_block["band"])
check("should_synthesize=False when faithfulness < 0.50", r_block["should_synthesize"] == False)

# T11: ConfidenceScorer faithfulness LOW gate (0.50-0.70)
r_low = scorer.score({
    "faithfulness": 0.65,
    "answer_relevance": 0.90,
    "context_precision": 0.88,
    "context_recall": 0.90,
    "citation_coverage": 1.00,
})
print(f"\n[T11] Faithfulness LOW gate (0.65): band={r_low['band']}, should_synthesize={r_low['should_synthesize']}")
check("band=LOW when faithfulness < 0.70", r_low["band"] == "LOW", r_low["band"])
check("should_synthesize=True when faithfulness >= 0.50", r_low["should_synthesize"] == True)
check("faithfulness_warning=True when 0.50 <= faithfulness < 0.70", r_low["faithfulness_warning"] == True)

# T12: ConfidenceScorer MEDIUM
r_med = scorer.score({
    "faithfulness": 0.75,
    "answer_relevance": 0.65,
    "context_precision": 0.60,
    "context_recall": 0.58,
    "citation_coverage": 0.75,
})
print(f"\n[T12] MEDIUM confidence: confidence={r_med['confidence']}, band={r_med['band']}")
check("band=MEDIUM for moderate metrics", r_med["band"] == "MEDIUM", r_med["band"])

# T13: Confidence formula correctness
# 0.35*0.9 + 0.25*0.8 + 0.15*0.7 + 0.15*0.7 + 0.10*1.0 = 0.315+0.200+0.105+0.105+0.100 = 0.825
r_formula = scorer.score({
    "faithfulness": 0.9,
    "answer_relevance": 0.8,
    "context_precision": 0.7,
    "context_recall": 0.7,
    "citation_coverage": 1.0,
})
expected = round(0.35*0.9 + 0.25*0.8 + 0.15*0.7 + 0.15*0.7 + 0.10*1.0, 4)
print(f"\n[T13] Confidence formula: expected={expected}, got={r_formula['confidence']}")
check(f"confidence formula correct (expected {expected})", abs(r_formula["confidence"] - expected) < 0.001, r_formula["confidence"])

# Summary
print(f"\n{'='*50}")
if errors:
    print(f"FAILED: {len(errors)} test(s): {errors}")
    sys.exit(1)
else:
    print(f"ALL {13} TESTS PASSED")
