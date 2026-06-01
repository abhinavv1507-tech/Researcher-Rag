"""
main.py — Single entry point for the RAG Research Agent.

Usage:
    python main.py --query "What methods do LLM agents use for tool selection?"
    python main.py --query "..." --verbose
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from utils.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced RAG Research Agent — query interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --query \"What methods do LLM agents use for tool selection?\"\n"
            "  python main.py --query \"Compare ReAct and Reflexion for multi-step reasoning\"\n"
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Research question to answer.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full answer + all intermediate state fields.",
    )
    args = parser.parse_args()

    log.info("main.start", query=args.query)

    try:
        from graph import run_query
        state = run_query(args.query)
    except Exception as e:
        log.error("main.failed", error=str(e))
        print(f"\n[ERROR] Pipeline failed: {e}")
        sys.exit(1)

    # Print answer
    print("\n" + "=" * 70)
    print("  RESEARCH ANSWER")
    print("=" * 70)
    print(f"\nQuery: {args.query}\n")
    print(state.get("answer", "(no answer generated)"))
    print()

    # Citations
    citations = state.get("citations", [])
    if citations:
        print("Citations:")
        for c in citations:
            print(f"  • arXiv:{c}")
    else:
        print("Citations: (none verified)")

    print()
    print("-" * 70)
    print(f"Escalated to graph:   {'Yes' if state.get('escalated') else 'No'}")
    print(f"Chunks used:          {len(state.get('reranked_chunks', []))}")
    print(f"Sub-queries executed: {len(state.get('sub_queries', []))}")
    print(f"Graph context items:  {len(state.get('graph_context', []))}")
    print("-" * 70)

    if args.verbose:
        import json
        print("\n[Verbose] Full state:")
        safe_state = {
            k: v for k, v in state.items()
            if k not in ("dense_results", "lexical_results")
        }
        print(json.dumps(safe_state, indent=2, default=str))

    log.info("main.done")


if __name__ == "__main__":
    main()
