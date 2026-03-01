"""Reranker node — cross-encoder precision re-ranking."""
from __future__ import annotations

import time

from app.graph.state import RAGState, Span
from app.retrieval.reranker import get_reranker
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def reranker_node(state: RAGState) -> dict:
    """Re-rank retrieved documents with a cross-encoder."""
    start = time.time() * 1000

    query = state.get("rewritten_query") or state["query"]
    documents = state.get("documents", [])
    top_k = get_settings().rerank_top_k

    if not documents:
        logger.warning("reranker_no_docs")
        span: Span = {
            "node": "reranker",
            "start_ms": start,
            "end_ms": time.time() * 1000,
            "metadata": {"reranked": 0},
        }
        return {"reranked_documents": [], "trace": state.get("trace", []) + [span]}

    reranker = get_reranker()
    reranked = reranker.rerank(query, documents, top_k=top_k)

    span = {
        "node": "reranker",
        "start_ms": start,
        "end_ms": time.time() * 1000,
        "metadata": {"input_docs": len(documents), "reranked": len(reranked)},
    }

    return {
        "reranked_documents": reranked,
        "trace": state.get("trace", []) + [span],
    }
