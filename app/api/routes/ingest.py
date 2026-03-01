"""PDF ingestion route."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.ingestion.loader import load_pdf
from app.ingestion.cleaner import clean_documents
from app.ingestion.chunker import chunk_documents
from app.vectorstore.faiss_store import get_faiss_store, reset_faiss_store
from app.observability.logger import log_ingest_request
from app.core.config import get_settings

router = APIRouter()


class IngestResponse(BaseModel):
    filename: str
    pages_extracted: int
    chunks_created: int
    total_index_size: int
    duration_ms: float


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    start = time.time()

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
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
        reset_faiss_store()  # force reload on next request

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
