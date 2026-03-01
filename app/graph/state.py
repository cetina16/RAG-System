"""LangGraph state definition for the RAG pipeline."""
from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.documents import Document


class Span(TypedDict):
    node: str
    start_ms: float
    end_ms: float
    metadata: dict[str, Any]


class RAGState(TypedDict):
    # Input
    query: str
    top_k: int

    # After query rewriting
    rewritten_query: str
    query_variants: list[str]

    # After retrieval
    documents: list[Document]

    # After re-ranking
    reranked_documents: list[Document]

    # After generation
    response: str
    citations: list[dict[str, Any]]

    # Observability
    trace: list[Span]
    cache_hit: bool

    # Error handling
    error: Optional[str]
