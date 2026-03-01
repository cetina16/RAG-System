"""FAISS vector store with persist/load support."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embeddings.embedder import Embedder, get_embedder

logger = get_logger(__name__)

_FAISS_FILE = "index.faiss"
_DOCS_FILE  = "docstore.pkl"
_META_FILE  = "meta.json"


class FAISSStore:
    """
    FAISS IndexFlatIP (inner product = cosine when vectors are normalized).

    Stores raw Documents alongside the index for retrieval.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self._index: faiss.IndexFlatIP | None = None
        self._documents: list[Document] = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[Document]) -> None:
        """Embed and add documents to the FAISS index."""
        if not docs:
            return

        texts = [d.page_content for d in docs]
        vectors = self.embedder.embed(texts, normalize=True)

        if self._index is None:
            dim = vectors.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            logger.info("faiss_index_created", dim=dim)

        self._index.add(vectors)
        self._documents.extend(docs)
        logger.info("faiss_docs_added", added=len(docs), total=len(self._documents))

    # ── Query ──────────────────────────────────────────────────────────────────

    def similarity_search(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        """
        Search for the k nearest documents.

        Returns list of (Document, score) sorted by descending similarity.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)
        q_vec = self.embedder.embed_query(query).reshape(1, -1)
        scores, indices = self._index.search(q_vec, k)

        results: list[tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._documents[idx], float(score)))
        return results

    # ── Persist / Load ─────────────────────────────────────────────────────────

    def save(self, directory: str | Path | None = None) -> None:
        directory = Path(directory or get_settings().faiss_index_path)
        directory.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            logger.warning("faiss_save_skipped", reason="empty index")
            return

        faiss.write_index(self._index, str(directory / _FAISS_FILE))

        with open(directory / _DOCS_FILE, "wb") as f:
            pickle.dump(self._documents, f)

        with open(directory / _META_FILE, "w") as f:
            json.dump({"total": len(self._documents)}, f)

        logger.info("faiss_saved", directory=str(directory), docs=len(self._documents))

    @classmethod
    def load(cls, directory: str | Path | None = None, embedder: Embedder | None = None) -> "FAISSStore":
        directory = Path(directory or get_settings().faiss_index_path)
        index_path = directory / _FAISS_FILE
        docs_path  = directory / _DOCS_FILE

        if not index_path.exists() or not docs_path.exists():
            logger.info("faiss_no_existing_index", directory=str(directory))
            return cls(embedder=embedder)

        store = cls(embedder=embedder)
        store._index = faiss.read_index(str(index_path))

        with open(docs_path, "rb") as f:
            store._documents = pickle.load(f)

        logger.info(
            "faiss_loaded",
            directory=str(directory),
            docs=len(store._documents),
            vectors=store._index.ntotal,
        )
        return store

    @property
    def total_documents(self) -> int:
        return len(self._documents)


_store_cache: FAISSStore | None = None


def get_faiss_store() -> FAISSStore:
    """Return the global FAISSStore, loading from disk if available."""
    global _store_cache
    if _store_cache is None:
        _store_cache = FAISSStore.load()
    return _store_cache


def reset_faiss_store() -> None:
    """Reset the cached store (used after ingestion)."""
    global _store_cache
    _store_cache = None
