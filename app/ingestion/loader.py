"""PDF loader using PyMuPDF for robust extraction from messy real-world documents."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from langchain_core.documents import Document

from app.core.logging import get_logger

logger = get_logger(__name__)


def _extract_title(doc: fitz.Document, source: str) -> str:
    """Try to get a meaningful title from PDF metadata or filename."""
    meta = doc.metadata or {}
    title = meta.get("title", "").strip()
    if title and len(title) > 3:
        return title
    return Path(source).stem


def load_pdf(path: str | Path) -> list[Document]:
    """
    Load a PDF file and return a list of LangChain Documents, one per page.

    Each Document carries metadata:
        source     : file path
        page       : 0-based page number
        page_label : human-readable page label (e.g. "iv", "12")
        title      : document title
        total_pages: total number of pages
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    documents: list[Document] = []

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.error("pdf_open_failed", path=str(path), error=str(exc))
        raise

    title = _extract_title(doc, str(path))
    total_pages = len(doc)

    logger.info("loading_pdf", path=str(path), pages=total_pages, title=title)

    for page_num, page in enumerate(doc):
        try:
            text = page.get_text("text")  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("page_extraction_failed", page=page_num, error=str(exc))
            text = ""

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "page": page_num,
                    "page_label": page.get_label() or str(page_num + 1),
                    "title": title,
                    "total_pages": total_pages,
                },
            )
        )

    doc.close()
    logger.info("pdf_loaded", path=str(path), extracted_pages=len(documents))
    return documents


def load_pdfs_from_dir(directory: str | Path) -> Iterator[Document]:
    """Yield Documents from all PDFs in a directory (non-recursive)."""
    directory = Path(directory)
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning("no_pdfs_found", directory=str(directory))
        return

    for pdf_path in pdf_files:
        try:
            yield from load_pdf(pdf_path)
        except Exception as exc:
            logger.error("pdf_skipped", path=str(pdf_path), error=str(exc))
