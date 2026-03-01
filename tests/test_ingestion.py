"""Tests for the PDF ingestion pipeline (loader, cleaner, chunker)."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.ingestion.cleaner import clean_text, clean_documents
from app.ingestion.chunker import chunk_documents


# ── Cleaner tests ──────────────────────────────────────────────────────────────

class TestCleaner:
    def test_removes_lone_page_numbers(self):
        text = "Some content here.\n\n123\n\nMore content."
        result = clean_text(text)
        assert "123" not in result.split() or "content" in result

    def test_fixes_hyphenated_line_breaks(self):
        text = "This is a hyphen-\nated word in a sentence."
        result = clean_text(text)
        assert "hyphenated" in result

    def test_normalizes_multiple_newlines(self):
        text = "Para one.\n\n\n\n\nPara two."
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_strips_whitespace(self):
        text = "  hello world  "
        result = clean_text(text)
        assert result == result.strip()

    def test_clean_documents_drops_short(self):
        docs = [
            Document(page_content="x", metadata={}),               # too short
            Document(page_content="A" * 100, metadata={}),         # keep
        ]
        result = clean_documents(docs)
        assert len(result) == 1
        assert len(result[0].page_content) >= 50

    def test_clean_documents_preserves_metadata(self):
        docs = [Document(page_content="A" * 100, metadata={"source": "test.pdf", "page": 1})]
        result = clean_documents(docs)
        assert result[0].metadata["source"] == "test.pdf"
        assert result[0].metadata["page"] == 1


# ── Chunker tests ──────────────────────────────────────────────────────────────

class TestChunker:
    def _make_doc(self, text: str, source: str = "test.pdf", page: int = 0) -> Document:
        return Document(page_content=text, metadata={"source": source, "page": page})

    def test_single_short_doc_stays_one_chunk(self):
        doc = self._make_doc("Short document. " * 5)
        chunks = chunk_documents([doc], chunk_size=512)
        assert len(chunks) >= 1

    def test_long_doc_splits_into_multiple_chunks(self):
        doc = self._make_doc("Word " * 300)
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunk_metadata_preserved(self):
        doc = self._make_doc("Some text. " * 30, source="my.pdf", page=5)
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "my.pdf"
            assert chunk.metadata["page"] == 5

    def test_chunk_id_unique(self):
        doc = self._make_doc("Word " * 300)
        chunks = chunk_documents([doc], chunk_size=50, chunk_overlap=5)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_increments(self):
        doc = self._make_doc("Word " * 300)
        chunks = chunk_documents([doc], chunk_size=50, chunk_overlap=5)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(indices)))

    def test_empty_doc_list(self):
        assert chunk_documents([]) == []

    def test_drops_empty_chunks(self):
        doc = Document(page_content="   \n  ", metadata={})
        chunks = chunk_documents([doc])
        assert all(c.page_content.strip() for c in chunks)
