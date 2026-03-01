"""Evaluation pipeline — runs retrieval against a labelled eval set."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.metrics import compute_all_metrics, mean_reciprocal_rank
from app.retrieval.hybrid_retriever import HybridRetriever
from app.vectorstore.faiss_store import get_faiss_store
from app.core.logging import get_logger

logger = get_logger(__name__)


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    """
    Load evaluation examples from a JSONL file.

    Each line must be JSON with keys:
        query:            str
        relevant_doc_ids: list[str]  (chunk_ids of ground-truth docs)
    """
    path = Path(path)
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run_evaluation(
    eval_path: str | Path,
    k: int = 5,
) -> dict[str, Any]:
    """
    Run the retrieval evaluation pipeline.

    Args:
        eval_path: path to JSONL eval set
        k:         top-k to evaluate at

    Returns:
        dict with per-query metrics and aggregate scores
    """
    examples = load_eval_set(eval_path)
    if not examples:
        return {"error": "Empty evaluation set"}

    store = get_faiss_store()
    retriever = HybridRetriever(store)

    per_query: list[dict[str, Any]] = []
    all_retrieved: list[list[str]] = []
    all_relevant:  list[set[str]]  = []

    for ex in examples:
        query       = ex["query"]
        relevant_ids = set(ex["relevant_doc_ids"])

        docs = retriever.retrieve(query, top_k=k)
        retrieved_ids = [d.metadata.get("chunk_id", "") for d in docs]

        metrics = compute_all_metrics(retrieved_ids, relevant_ids, k=k)
        per_query.append({"query": query, **metrics})

        all_retrieved.append(retrieved_ids)
        all_relevant.append(relevant_ids)

    # Aggregate
    agg: dict[str, float] = {}
    metric_keys = [key for key in per_query[0] if key != "query"]
    for key in metric_keys:
        agg[key] = round(
            sum(q[key] for q in per_query) / len(per_query), 4
        )

    logger.info("evaluation_complete", queries=len(examples), k=k, **agg)

    return {
        "summary":   agg,
        "per_query": per_query,
        "k":         k,
        "queries":   len(examples),
    }
