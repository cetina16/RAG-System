#!/usr/bin/env python3
"""
CLI script to ingest PDF files into the RAG system's FAISS index.

Usage:
    python scripts/ingest_pdfs.py --dir data/pdfs
    python scripts/ingest_pdfs.py --file path/to/document.pdf
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from app.core.logging import configure_logging
from app.ingestion.loader import load_pdf, load_pdfs_from_dir
from app.ingestion.cleaner import clean_documents
from app.ingestion.chunker import chunk_documents
from app.vectorstore.faiss_store import FAISSStore, get_embedder


def ingest_file(pdf_path: Path, store: FAISSStore) -> tuple[int, int]:
    """Ingest a single PDF. Returns (pages, chunks)."""
    raw_docs   = load_pdf(pdf_path)
    clean_docs = clean_documents(raw_docs)
    chunks     = chunk_documents(clean_docs)
    store.add_documents(chunks)
    return len(raw_docs), len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into RAG FAISS index")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir",  type=Path, help="Directory containing PDF files")
    group.add_argument("--file", type=Path, help="Single PDF file to ingest")
    parser.add_argument("--chunk-size",    type=int, default=512, help="Chunk size (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=64,  help="Chunk overlap (default: 64)")
    parser.add_argument("--log-level",     default="INFO", help="Log level")
    args = parser.parse_args()

    configure_logging(args.log_level)

    # Build a fresh store (load existing index if present)
    from app.vectorstore.faiss_store import FAISSStore
    store = FAISSStore.load()

    total_pages  = 0
    total_chunks = 0
    start        = time.time()

    if args.file:
        pdf_files = [args.file]
    else:
        pdf_files = sorted(args.dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {args.dir}")
            sys.exit(1)

    print(f"Ingesting {len(pdf_files)} PDF file(s)...")

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        try:
            pages, chunks = ingest_file(pdf_path, store)
            total_pages  += pages
            total_chunks += chunks
            tqdm.write(f"  {pdf_path.name}: {pages} pages → {chunks} chunks")
        except Exception as exc:
            tqdm.write(f"  ERROR {pdf_path.name}: {exc}")

    # Persist
    store.save()
    duration = time.time() - start

    print(f"\n{'='*50}")
    print(f"Ingestion complete in {duration:.1f}s")
    print(f"  Files:  {len(pdf_files)}")
    print(f"  Pages:  {total_pages}")
    print(f"  Chunks: {total_chunks}")
    print(f"  Index size: {store.total_documents} documents")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
