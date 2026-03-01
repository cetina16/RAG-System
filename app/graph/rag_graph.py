"""LangGraph StateGraph definition for the RAG pipeline."""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.graph.state import RAGState
from app.graph.nodes.query_rewriter import query_rewriter_node
from app.graph.nodes.retriever_node import retriever_node
from app.graph.nodes.reranker_node import reranker_node
from app.graph.nodes.generator_node import generator_node
from app.core.logging import get_logger

logger = get_logger(__name__)


def _should_continue_after_retrieval(state: RAGState) -> str:
    """Conditional edge: if no documents retrieved, skip to generator (handles gracefully)."""
    if not state.get("documents"):
        logger.warning("no_documents_retrieved", query=state["query"])
        return "generator"
    return "reranker"


def build_rag_graph() -> StateGraph:
    """Build and compile the LangGraph RAG pipeline."""
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("retriever",      retriever_node)
    graph.add_node("reranker",       reranker_node)
    graph.add_node("generator",      generator_node)

    # Edges
    graph.add_edge(START, "query_rewriter")
    graph.add_edge("query_rewriter", "retriever")

    # Conditional: skip reranker if no docs found
    graph.add_conditional_edges(
        "retriever",
        _should_continue_after_retrieval,
        {"reranker": "reranker", "generator": "generator"},
    )

    graph.add_edge("reranker",  "generator")
    graph.add_edge("generator", END)

    compiled = graph.compile()
    logger.info("rag_graph_compiled")
    return compiled


# Module-level compiled graph (singleton)
_graph_cache = None


def get_rag_graph():
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = build_rag_graph()
    return _graph_cache


def run_rag_pipeline(query: str, top_k: int = 20) -> RAGState:
    """
    Execute the full RAG pipeline for a query.

    Args:
        query: user's natural language query
        top_k: number of documents to retrieve in first stage

    Returns:
        Final RAGState with response, citations, and trace
    """
    graph = get_rag_graph()

    initial_state: RAGState = {
        "query":              query,
        "top_k":              top_k,
        "rewritten_query":    "",
        "query_variants":     [],
        "documents":          [],
        "reranked_documents": [],
        "response":           "",
        "citations":          [],
        "trace":              [],
        "cache_hit":          False,
        "error":              None,
    }

    final_state = graph.invoke(initial_state)
    return final_state
