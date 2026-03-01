"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.vectorstore.faiss_store import get_faiss_store
from app.graph.rag_graph import get_rag_graph
from app.api.routes import health, ingest, query as query_route, documents

logger = get_logger(__name__)

# Rate limiter — keyed by client IP
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-load models and index. Shutdown: nothing to clean up."""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info("startup_begin")
    store = get_faiss_store()
    logger.info("faiss_ready", docs=store.total_documents)

    get_rag_graph()
    logger.info("langgraph_ready")

    yield

    logger.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG System API",
        description=(
            "Production-ready Retrieval-Augmented Generation system "
            "with hybrid retrieval, LangGraph pipeline, and Mistral LLM."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files (frontend)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # API routes
    prefix = "/api/v1"
    app.include_router(health.router,       prefix=prefix, tags=["Health"])
    app.include_router(ingest.router,       prefix=prefix, tags=["Ingestion"])
    app.include_router(query_route.router,  prefix=prefix, tags=["Query"])
    app.include_router(documents.router,    prefix=prefix, tags=["Documents"])

    # Serve frontend at root
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def frontend(request: Request):
        return (static_dir / "index.html").read_text()

    return app


app = create_app()
