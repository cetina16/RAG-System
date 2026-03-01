"""PDF ingestion route."""
from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.ingestion.loader import load_pdf
from app.ingestion.cleaner import clean_documents
from app.ingestion.chunker import chunk_documents
from app.vectorstore.faiss_store import get_faiss_store, reset_faiss_store
from app.observability.logger import log_ingest_request

router  = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# In-memory set of MD5 hashes of already-ingested PDFs (survives hot-reload)
_ingested_hashes: set[str] = set()


class IngestResponse(BaseModel):
    filename: str
    pages_extracted: int
    chunks_created: int
    total_index_size: int
    duration_ms: float
    duplicate: bool = False


@router.post("/ingest", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest_pdf(request: Request, file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    start   = time.time()
    content = await file.read()

    # ── Duplicate detection ────────────────────────────────────────────────
    content_hash = hashlib.md5(content).hexdigest()
    if content_hash in _ingested_hashes:
        raise HTTPException(
            status_code=409,
            detail=f"'{file.filename}' has already been ingested (duplicate content)."
        )

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Ingestion pipeline
        raw_docs = load_pdf(tmp_path)

        # Replace temp path with the original filename in all metadata
        for doc in raw_docs:
            doc.metadata["source"] = file.filename
            doc.metadata["chunk_id"] = doc.metadata.get("chunk_id", "").replace(
                str(tmp_path), file.filename
            )

        clean_docs = clean_documents(raw_docs)
        chunks     = chunk_documents(clean_docs)

        # Embed + store
        store = get_faiss_store()
        store.add_documents(chunks)
        store.save()
        reset_faiss_store()

        # Mark as ingested only after success
        _ingested_hashes.add(content_hash)

        duration_ms = (time.time() - start) * 1000
        log_ingest_request(
            source=file.filename,
            pages=len(raw_docs),
            chunks=len(chunks),
            duration_ms=duration_ms,
        )

        return IngestResponse(
            filename=file.filename,
            pages_extracted=len(raw_docs),
            chunks_created=len(chunks),
            total_index_size=get_faiss_store().total_documents,
            duration_ms=round(duration_ms, 1),
        )
    finally:
        tmp_path.unlink(missing_ok=True)
