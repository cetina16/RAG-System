"""Retriever node — runs hybrid BM25 + FAISS search."""
from __future__ import annotations

import time

from app.graph.state import RAGState, Span
from app.retrieval.hybrid_retriever import HybridRetriever
from app.vectorstore.faiss_store import get_faiss_store
from app.core.logging import get_logger

logger = get_logger(__name__)


def retriever_node(state: RAGState) -> dict:
    """Retrieve documents using hybrid retrieval over all query variants."""
    start = time.time() * 1000

    query = state.get("rewritten_query") or state["query"]
    variants = state.get("query_variants", [query])
    top_k = state.get("top_k", 20)

    store = get_faiss_store()
    retriever = HybridRetriever(store)

    # Retrieve for each variant and merge by chunk_id (dedup)
    seen_ids: set[str] = set()
    merged_docs = []

    for variant in variants:
        docs = retriever.retrieve(variant, top_k=top_k)
        for doc in docs:
            cid = doc.metadata.get("chunk_id", doc.page_content[:40])
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged_docs.append(doc)

    # Sort merged by rrf_score descending
    merged_docs.sort(key=lambda d: d.metadata.get("rrf_score", 0.0), reverse=True)
    merged_docs = merged_docs[:top_k]

    logger.info(
        "retrieval_done",
        variants=len(variants),
        merged_docs=len(merged_docs),
    )

    span: Span = {
        "node": "retriever",
        "start_ms": start,
        "end_ms": time.time() * 1000,
        "metadata": {"retrieved": len(merged_docs)},
    }

    return {
        "documents": merged_docs,
        "trace": state.get("trace", []) + [span],
    }
