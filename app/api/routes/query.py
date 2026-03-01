"""Query route with optional SSE streaming."""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.graph.rag_graph import run_rag_pipeline
from app.cache.query_cache import cache_get, cache_set
from app.observability.logger import log_query_request
from app.observability.tracer import summarize_trace
from app.llm.mistral_client import get_llm_client
from app.core.config import get_settings

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=20, ge=1, le=100)
    stream: bool = False


class Citation(BaseModel):
    doc_index: int
    source: str
    page: int | None
    title: str | None
    snippet: str
    rerank_score: float | None


class QueryResponse(BaseModel):
    query: str
    rewritten_query: str
    response: str
    citations: list[Citation]
    cache_hit: bool
    trace: dict


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if request.stream:
        return await _streaming_query(request)
    return await _standard_query(request)


async def _standard_query(request: QueryRequest) -> QueryResponse:
    # Check cache
    cached = cache_get(request.query, request.top_k)
    if cached is not None:
        cached["cache_hit"] = True
        return QueryResponse(**cached)

    state = run_rag_pipeline(query=request.query, top_k=request.top_k)

    log_query_request(request.query, state, cache_hit=False)

    result = {
        "query":            request.query,
        "rewritten_query":  state.get("rewritten_query", ""),
        "response":         state.get("response", ""),
        "citations":        state.get("citations", []),
        "cache_hit":        False,
        "trace":            summarize_trace(state.get("trace", [])),
    }

    cache_set(request.query, request.top_k, result)
    return QueryResponse(**result)


async def _streaming_query(request: QueryRequest) -> StreamingResponse:
    """SSE streaming response using the LLM's stream() method."""
    from app.graph.nodes.query_rewriter import query_rewriter_node
    from app.graph.nodes.retriever_node import retriever_node
    from app.graph.nodes.reranker_node import reranker_node
    from app.graph.nodes.generator_node import _build_context, _extract_citations, _GENERATION_PROMPT
    from app.graph.state import RAGState

    async def event_stream() -> AsyncIterator[str]:
        # Run pipeline up to generator
        initial: RAGState = {
            "query":              request.query,
            "top_k":              request.top_k,
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

        state = {**initial, **query_rewriter_node(initial)}
        state = {**state,   **retriever_node(state)}       # type: ignore[arg-type]
        state = {**state,   **reranker_node(state)}        # type: ignore[arg-type]

        docs  = state.get("reranked_documents") or state.get("documents", [])
        citations = _extract_citations(docs)

        # Emit metadata event
        yield f"data: {json.dumps({'type': 'meta', 'citations': citations, 'rewritten_query': state.get('rewritten_query', '')})}\n\n"

        if docs:
            context = _build_context(docs)
            prompt  = _GENERATION_PROMPT.format(context=context, query=request.query)
            llm     = get_llm_client()
            for token in llm.stream(prompt):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        else:
            fallback = "No relevant documents found for your query."
            yield f"data: {json.dumps({'type': 'token', 'content': fallback})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
