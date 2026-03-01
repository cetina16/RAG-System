"""Health check route."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.vectorstore.faiss_store import get_faiss_store
from app.cache.query_cache import cache_stats

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    index_documents: int
    cache: dict


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    store = get_faiss_store()
    return HealthResponse(
        status="ok",
        index_documents=store.total_documents,
        cache=cache_stats(),
    )
