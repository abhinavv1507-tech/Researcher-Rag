"""
evaluate.py — Stage 5: CLI Evaluation Harness

Usage:
    python evaluate.py --testset tests.jsonl

Each line of the testset file must be a JSON object: {"question": "..."}

Outputs results.jsonl and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from utils.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


def _count_tool_calls(state: dict) -> int:
    """Estimate tool call count from sub-queries."""
    return len(state.get("sub_queries", []))


def run_evaluate(testset_path: Path, output_path: Path = Path("results.jsonl")) -> None:
    from graph import run_query  # lazy import so graph isn't loaded until needed

    questions: list[str] = []
    with open(testset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question") or obj.get("query") or obj.get("q")
            if q:
                questions.append(q)

    if not questions:
        log.error("evaluate.no_questions", path=str(testset_path))
        sys.exit(1)

    log.info("evaluate.start", n_questions=len(questions))
    records: list[dict] = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, question in enumerate(questions, 1):
            log.info("evaluate.running", question_num=i, total=len(questions), query=question[:80])
            t0 = time.perf_counter()
            try:
                state = run_query(question)
                latency = time.perf_counter() - t0
                record = {
                    "query": question,
                    "answer": state.get("answer", ""),
                    "citations": state.get("citations", []),
                    "latency_seconds": round(latency, 3),
                    "tool_call_count": _count_tool_calls(state),
                    "escalated": state.get("escalated", False),
                    "chunks_used": len(state.get("reranked_chunks", [])),
                }
            except Exception as e:
                latency = time.perf_counter() - t0
                log.error("evaluate.error", question=question[:80], error=str(e))
                record = {
                    "query": question,
                    "answer": f"ERROR: {e}",
                    "citations": [],
                    "latency_seconds": round(latency, 3),
                    "tool_call_count": 0,
                    "escalated": False,
                    "chunks_used": 0,
                }

            records.append(record)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    _print_summary(records)
    log.info("evaluate.done", output=str(output_path))


def _print_summary(records: list[dict]) -> None:
    n = len(records)
    if n == 0:
        print("No results to summarise.")
        return

    mean_latency = sum(r["latency_seconds"] for r in records) / n
    mean_tools = sum(r["tool_call_count"] for r in records) / n
    escalation_rate = sum(1 for r in records if r["escalated"]) / n
    mean_chunks = sum(r["chunks_used"] for r in records) / n

    # ASCII table
    col_w = [40, 14, 12, 16, 13]
    header = ["Metric", "Mean Latency", "Mean Tools", "Escalation Rate", "Mean Chunks"]
    values = [
        "Summary",
        f"{mean_latency:.2f}s",
        f"{mean_tools:.1f}",
        f"{escalation_rate:.1%}",
        f"{mean_chunks:.1f}",
    ]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    row_fmt = "|" + "|".join(f"{{:<{w}}}" for w in col_w) + "|"

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(sep)
    print(row_fmt.format(*header))
    print(sep)
    print(row_fmt.format(*values))
    print(sep)
    print(f"\nTotal questions: {n}")

    # Per-question mini-table
    print("\n" + "-" * 80)
    print(f"{'#':<4} {'Query':<42} {'Lat(s)':<8} {'Chunks':<8} {'Esc':<5}")
    print("-" * 80)
    for i, r in enumerate(records, 1):
        q_short = r["query"][:40]
        print(
            f"{i:<4} {q_short:<42} {r['latency_seconds']:<8.2f} {r['chunks_used']:<8} {'Y' if r['escalated'] else 'N':<5}"
        )
    print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI evaluation harness for the RAG research agent."
    )
    parser.add_argument(
        "--testset",
        type=Path,
        required=True,
        help="Path to a .jsonl file where each line is {\"question\": \"...\"}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.jsonl"),
        help="Output path for results (default: results.jsonl)",
    )
    args = parser.parse_args()

    if not args.testset.exists():
        print(f"ERROR: testset file not found: {args.testset}")
        sys.exit(1)

    run_evaluate(args.testset, args.output)


if __name__ == "__main__":
    main()
