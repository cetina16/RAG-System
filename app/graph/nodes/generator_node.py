"""Generator node — builds grounded response with citations."""
from __future__ import annotations

import time
from typing import Any

from langchain_core.documents import Document

from app.graph.state import RAGState, Span
from app.llm.mistral_client import get_llm_client
from app.core.logging import get_logger

logger = get_logger(__name__)

_GENERATION_PROMPT = """\
You are a helpful assistant that answers questions based ONLY on the provided context.
Always cite the source documents you used by referencing [Doc N] in your answer.
If the context does not contain sufficient information, say so clearly.

Context:
{context}

Question: {query}

Answer (cite sources as [Doc N]):"""


def _build_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(
            f"[Doc {i}] (source: {source}, page: {page})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_citations(docs: list[Document]) -> list[dict[str, Any]]:
    citations = []
    for i, doc in enumerate(docs, start=1):
        citations.append({
            "doc_index": i,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "page_label": doc.metadata.get("page_label"),
            "title": doc.metadata.get("title"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "rerank_score": doc.metadata.get("rerank_score"),
            "snippet": doc.page_content[:200],
        })
    return citations


def generator_node(state: RAGState) -> dict:
    """Generate a grounded response from re-ranked documents."""
    start = time.time() * 1000

    query = state["query"]
    docs  = state.get("reranked_documents") or state.get("documents", [])

    if not docs:
        span: Span = {
            "node": "generator",
            "start_ms": start,
            "end_ms": time.time() * 1000,
            "metadata": {"context_docs": 0},
        }
        return {
            "response": (
                "I could not find relevant information in the knowledge base to answer your question. "
                "Please try rephrasing or uploading relevant documents."
            ),
            "citations": [],
            "trace": state.get("trace", []) + [span],
        }

    context = _build_context(docs)
    prompt  = _GENERATION_PROMPT.format(context=context, query=query)

    try:
        llm      = get_llm_client()
        response = llm.generate(prompt)
    except Exception as exc:
        logger.error("generation_failed", error=str(exc))
        response = f"Generation failed: {exc}"

    citations = _extract_citations(docs)

    logger.info(
        "generation_done",
        query_preview=query[:60],
        context_docs=len(docs),
        response_len=len(response),
    )

    span = {
        "node": "generator",
        "start_ms": start,
        "end_ms": time.time() * 1000,
        "metadata": {"context_docs": len(docs), "response_len": len(response)},
    }

    return {
        "response": response,
        "citations": citations,
        "trace": state.get("trace", []) + [span],
    }
