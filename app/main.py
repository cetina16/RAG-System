"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.vectorstore.faiss_store import get_faiss_store
from app.graph.rag_graph import get_rag_graph
from app.api.routes import health, ingest, query as query_route

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-load models and index. Shutdown: nothing to clean up."""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info("startup_begin")
    # Warm up FAISS store (loads from disk if exists)
    store = get_faiss_store()
    logger.info("faiss_ready", docs=store.total_documents)

    # Pre-compile the LangGraph (builds node graph once)
    get_rag_graph()
    logger.info("langgraph_ready")

    yield

    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG System API",
        description=(
            "Production-ready Retrieval-Augmented Generation system "
            "with hybrid retrieval, LangGraph pipeline, and Mistral LLM."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = "/api/v1"
    app.include_router(health.router,       prefix=prefix, tags=["Health"])
    app.include_router(ingest.router,       prefix=prefix, tags=["Ingestion"])
    app.include_router(query_route.router,  prefix=prefix, tags=["Query"])

    return app


app = create_app()
