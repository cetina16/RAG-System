"""Cross-encoder re-ranker using sentence-transformers."""
from __future__ import annotations

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker_cache: "Reranker | None" = None


class Reranker:
    """Scores (query, passage) pairs with a cross-encoder for precise re-ranking."""

    def __init__(self, model_name: str = _CROSS_ENCODER_MODEL) -> None:
        logger.info("loading_reranker", model=model_name)
        self._model = CrossEncoder(model_name)
        logger.info("reranker_ready", model=model_name)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Re-rank documents by cross-encoder score.

        Args:
            query:     user query
            documents: candidate documents from first-stage retrieval
            top_k:     number of documents to return (defaults to settings.rerank_top_k)

        Returns:
            top_k Documents sorted by cross-encoder score descending
        """
        settings = get_settings()
        top_k = top_k or settings.rerank_top_k

        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        scored = sorted(
            zip(documents, scores), key=lambda x: float(x[1]), reverse=True
        )

        results: list[Document] = []
        for doc, score in scored[:top_k]:
            reranked_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": round(float(score), 4)},
            )
            results.append(reranked_doc)

        logger.info(
            "reranking_done",
            input_docs=len(documents),
            output_docs=len(results),
            top_score=round(float(scored[0][1]), 4) if scored else None,
        )
        return results


def get_reranker() -> "Reranker":
    global _reranker_cache
    if _reranker_cache is None:
        _reranker_cache = Reranker()
    return _reranker_cache
