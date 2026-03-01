"""Hybrid retrieval: BM25 (keyword) + FAISS (semantic) fused via Reciprocal Rank Fusion."""
from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.logging import get_logger
from app.vectorstore.faiss_store import FAISSStore

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: each list contains chunk_ids in ranked order (best first)
        k:            RRF constant (typically 60)

    Returns:
        list of (chunk_id, rrf_score) sorted descending
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """
    Combines BM25 keyword search and FAISS semantic search.

    BM25 index is built at query time over the current FAISS store's documents.
    For production with very large corpora, pre-build and cache the BM25 index.
    """

    def __init__(self, store: FAISSStore) -> None:
        self._store = store
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[Document] = []

    def _ensure_bm25(self) -> None:
        docs = self._store._documents
        if self._bm25 is None or len(docs) != len(self._bm25_docs):
            corpus = [_tokenize(d.page_content) for d in docs]
            self._bm25 = BM25Okapi(corpus)
            self._bm25_docs = list(docs)
            logger.info("bm25_index_rebuilt", docs=len(docs))

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """
        Retrieve documents using hybrid search (BM25 + FAISS → RRF).

        Args:
            query: user query string
            top_k: number of results to return (defaults to settings.retrieval_top_k)

        Returns:
            list of Documents ordered by relevance
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k

        if self._store.total_documents == 0:
            logger.warning("retriever_empty_store")
            return []

        self._ensure_bm25()

        # ── BM25 retrieval ────────────────────────────────────────────────────
        bm25_scores = self._bm25.get_scores(_tokenize(query))  # type: ignore[union-attr]
        bm25_top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_k]
        bm25_ids = [
            self._bm25_docs[i].metadata.get("chunk_id", str(i))
            for i in bm25_top_indices
        ]

        # ── FAISS semantic retrieval ──────────────────────────────────────────
        faiss_results = self._store.similarity_search(query, k=top_k)
        faiss_ids = [
            doc.metadata.get("chunk_id", str(i))
            for i, (doc, _) in enumerate(faiss_results)
        ]

        # ── RRF fusion ────────────────────────────────────────────────────────
        fused = _reciprocal_rank_fusion([bm25_ids, faiss_ids], k=settings.rrf_k)

        # Build id → Document lookup
        id_to_doc: dict[str, Document] = {}
        for i, doc in enumerate(self._bm25_docs):
            cid = doc.metadata.get("chunk_id", str(i))
            id_to_doc[cid] = doc
        for doc, _ in faiss_results:
            cid = doc.metadata.get("chunk_id", "")
            if cid:
                id_to_doc[cid] = doc

        results: list[Document] = []
        for chunk_id, score in fused[:top_k]:
            if chunk_id in id_to_doc:
                doc = id_to_doc[chunk_id]
                # Attach retrieval score to metadata for observability
                enriched = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rrf_score": round(score, 6)},
                )
                results.append(enriched)

        logger.info(
            "hybrid_retrieval_done",
            query_preview=query[:60],
            bm25_hits=len(bm25_ids),
            faiss_hits=len(faiss_ids),
            fused_results=len(results),
        )
        return results
