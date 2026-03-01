"""Text cleaning utilities for messy real-world PDF content."""
from __future__ import annotations

import re
import unicodedata

import ftfy
from langchain_core.documents import Document


# Patterns that often represent repeating headers/footers in PDFs
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),                # lone page numbers
    re.compile(r"^\s*confidential\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*draft\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*www\.[^\s]+\s*$", re.IGNORECASE | re.MULTILINE),  # lone URLs
]

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE   = re.compile(r"[ \t]{2,}")
_HYPHEN_BREAK  = re.compile(r"-\n(\w)")   # hyphenation across line breaks


def _remove_boilerplate(text: str) -> str:
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return text


def _fix_unicode(text: str) -> str:
    """Use ftfy to fix mojibake and normalize to NFC."""
    text = ftfy.fix_text(text)
    return unicodedata.normalize("NFC", text)


def _rejoin_hyphenated(text: str) -> str:
    """Re-join words broken across lines with a hyphen."""
    return _HYPHEN_BREAK.sub(r"\1", text)


def _normalize_whitespace(text: str) -> str:
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single text string."""
    text = _fix_unicode(text)
    text = _rejoin_hyphenated(text)
    text = _remove_boilerplate(text)
    text = _normalize_whitespace(text)
    return text


def clean_document(doc: Document) -> Document:
    """Return a new Document with cleaned page_content."""
    cleaned = clean_text(doc.page_content)
    return Document(page_content=cleaned, metadata=doc.metadata.copy())


def clean_documents(docs: list[Document]) -> list[Document]:
    """Clean a list of Documents, dropping empty ones."""
    cleaned = [clean_document(d) for d in docs]
    return [d for d in cleaned if len(d.page_content.strip()) > 50]
