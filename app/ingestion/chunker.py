"""Smart chunking with metadata preservation."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import get_logger

logger = get_logger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    Split documents into overlapping chunks.

    Metadata from the parent document is propagated to every chunk.
    An additional `chunk_index` field tracks position within the source page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[Document] = []
    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        for idx, split_text in enumerate(splits):
            if not split_text.strip():
                continue
            chunk_meta = doc.metadata.copy()
            chunk_meta["chunk_index"] = idx
            chunk_meta["chunk_id"] = (
                f"{chunk_meta.get('source', 'unknown')}"
                f"::p{chunk_meta.get('page', 0)}"
                f"::c{idx}"
            )
            chunks.append(Document(page_content=split_text, metadata=chunk_meta))

    logger.info(
        "chunking_complete",
        input_docs=len(docs),
        output_chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunks
