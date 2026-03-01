"""Documents listing and index management routes."""
from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.vectorstore.faiss_store import get_faiss_store, reset_faiss_store
from app.api.routes.ingest import _ingested_hashes
from app.core.config import get_settings

router = APIRouter()


class FileInfo(BaseModel):
    filename: str
    chunks: int
    pages: list[int]


class DocumentsResponse(BaseModel):
    files: list[FileInfo]
    total_chunks: int


class ResetResponse(BaseModel):
    message: str


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents() -> DocumentsResponse:
    """Return a list of unique files currently in the FAISS index."""
    store = get_faiss_store()
    grouped: dict[str, dict] = defaultdict(lambda: {"chunks": 0, "pages": set()})

    for doc in store._documents:
        src = doc.metadata.get("source", "unknown")
        grouped[src]["chunks"] += 1
        page = doc.metadata.get("page")
        if page is not None:
            grouped[src]["pages"].add(page)

    files = [
        FileInfo(
            filename=src,
            chunks=info["chunks"],
            pages=sorted(info["pages"]),
        )
        for src, info in sorted(grouped.items())
    ]

    return DocumentsResponse(files=files, total_chunks=store.total_documents)


@router.delete("/index", response_model=ResetResponse)
async def reset_index() -> ResetResponse:
    """Delete the FAISS index from disk and reset the in-memory store."""
    index_path = Path(get_settings().faiss_index_path)
    if index_path.exists():
        shutil.rmtree(index_path)
    reset_faiss_store()
    _ingested_hashes.clear()   # allow re-ingesting the same files after a reset
    get_faiss_store()
    return ResetResponse(message="Index reset successfully.")
