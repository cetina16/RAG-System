"""Query rewriter node — expands the query for better retrieval coverage."""
from __future__ import annotations

import time

from app.graph.state import RAGState, Span
from app.llm.mistral_client import get_llm_client
from app.core.logging import get_logger

logger = get_logger(__name__)

_REWRITE_PROMPT = """\
You are a query optimization assistant for a document retrieval system.
Given the user's question, generate 2 alternative search queries that:
- Use different vocabulary and synonyms
- Target different aspects of the question
- Remain factually equivalent

Original question: {query}

Respond with ONLY the two alternative queries, one per line. No numbering, no explanations.
"""


def query_rewriter_node(state: RAGState) -> dict:
    """Rewrite the user query to improve retrieval recall."""
    start = time.time() * 1000
    query = state["query"]

    try:
        llm = get_llm_client()
        prompt = _REWRITE_PROMPT.format(query=query)
        raw = llm.generate(prompt)

        variants = [
            line.strip()
            for line in raw.strip().splitlines()
            if line.strip() and not line.startswith("[STUB")
        ][:2]

        # Always include the original query
        all_variants = [query] + variants
        # Use the first variant as the primary rewritten query (or original)
        rewritten = variants[0] if variants else query

        logger.info(
            "query_rewritten",
            original=query,
            variants=variants,
        )
    except Exception as exc:
        logger.warning("query_rewrite_failed", error=str(exc))
        rewritten = query
        all_variants = [query]

    span: Span = {
        "node": "query_rewriter",
        "start_ms": start,
        "end_ms": time.time() * 1000,
        "metadata": {"variants_count": len(all_variants)},
    }

    return {
        "rewritten_query": rewritten,
        "query_variants": all_variants,
        "trace": state.get("trace", []) + [span],
    }
