#!/usr/bin/env python3
"""
CLI script to run retrieval evaluation against a labelled eval set.

Eval set format (JSONL, one example per line):
    {"query": "What is RAG?", "relevant_doc_ids": ["path/to/doc.pdf::p0::c2"]}

Usage:
    python scripts/evaluate.py --eval-set data/eval.jsonl --k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import configure_logging
from app.evaluation.evaluator import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation")
    parser.add_argument("--eval-set", type=Path, required=True, help="Path to JSONL eval set")
    parser.add_argument("--k",        type=int, default=5,    help="Top-k to evaluate at")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    configure_logging(args.log_level)

    if not args.eval_set.exists():
        print(f"Eval set not found: {args.eval_set}")
        sys.exit(1)

    print(f"Running evaluation: {args.eval_set} (k={args.k})")
    results = run_evaluation(args.eval_set, k=args.k)

    if "error" in results:
        print(f"Error: {results['error']}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Evaluation Results (k={results['k']}, n={results['queries']} queries)")
    print(f"{'='*50}")
    for metric, value in results["summary"].items():
        print(f"  {metric:<20} {value:.4f}")
    print(f"{'='*50}")

    print("\nPer-query breakdown:")
    for i, q in enumerate(results["per_query"], 1):
        print(f"\n  [{i}] {q['query'][:60]}")
        for key, val in q.items():
            if key != "query":
                print(f"       {key:<20} {val:.4f}")


if __name__ == "__main__":
    main()
