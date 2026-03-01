"""Retrieval evaluation metrics: precision@k, recall@k, MRR, hit_rate."""
from __future__ import annotations


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents found in top-k results."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / rank_of_first_relevant_doc (0 if none found)."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    all_retrieved: list[list[str]],
    all_relevant: list[set[str]],
) -> float:
    """MRR over multiple queries."""
    if not all_retrieved:
        return 0.0
    rr_scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    return sum(rr_scores) / len(rr_scores)


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1 if at least one relevant document is in top-k, else 0."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & relevant_ids else 0.0


def compute_all_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> dict[str, float]:
    """Compute all metrics for a single query."""
    return {
        f"precision@{k}": round(precision_at_k(retrieved_ids, relevant_ids, k), 4),
        f"recall@{k}":    round(recall_at_k(retrieved_ids, relevant_ids, k), 4),
        "mrr":             round(reciprocal_rank(retrieved_ids, relevant_ids), 4),
        f"hit_rate@{k}":  round(hit_rate_at_k(retrieved_ids, relevant_ids, k), 4),
    }
