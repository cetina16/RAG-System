"""Structured request logger for RAG pipeline operations."""
from __future__ import annotations

import time
from typing import Any

from app.core.logging import get_logger
from app.observability.tracer import summarize_trace

logger = get_logger("rag.request")


def log_query_request(
    query: str,
    state: dict[str, Any],
    cache_hit: bool = False,
) -> None:
    """Log a completed RAG query with full observability metadata."""
    trace_summary = summarize_trace(state.get("trace", []))

    logger.info(
        "rag_query",
        query=query,
        cache_hit=cache_hit,
        retrieved_docs=len(state.get("documents", [])),
        reranked_docs=len(state.get("reranked_documents", [])),
        response_len=len(state.get("response", "")),
        citations=len(state.get("citations", [])),
        rewritten_query=state.get("rewritten_query", ""),
        **trace_summary,
    )


def log_ingest_request(
    source: str,
    pages: int,
    chunks: int,
    duration_ms: float,
) -> None:
    logger.info(
        "rag_ingest",
        source=source,
        pages=pages,
        chunks=chunks,
        duration_ms=round(duration_ms, 1),
    )
