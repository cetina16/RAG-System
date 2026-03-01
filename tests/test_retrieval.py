"""Tests for hybrid retrieval and re-ranking components."""
from __future__ import annotations

import numpy as np
import pytest
from langchain_core.documents import Document
from unittest.mock import MagicMock, patch

from app.retrieval.hybrid_retriever import _reciprocal_rank_fusion, HybridRetriever


# ── RRF Tests ──────────────────────────────────────────────────────────────────

class TestRRF:
    def test_single_list(self):
        result = _reciprocal_rank_fusion([["a", "b", "c"]])
        ids = [r[0] for r in result]
        assert ids == ["a", "b", "c"]

    def test_two_lists_agreement_scores_higher(self):
        # "a" appears first in both lists → should win
        result = _reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
        scores = {r[0]: r[1] for r in result}
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]

    def test_deduplication(self):
        result = _reciprocal_rank_fusion([["a", "a", "b"], ["a", "b"]])
        ids = [r[0] for r in result]
        assert ids.count("a") == 1

    def test_empty_lists(self):
        result = _reciprocal_rank_fusion([[], []])
        assert result == []

    def test_rrf_k_parameter(self):
        r1 = _reciprocal_rank_fusion([["a"]], k=1)
        r2 = _reciprocal_rank_fusion([["a"]], k=100)
        # higher k → lower score
        assert r1[0][1] > r2[0][1]


# ── HybridRetriever Tests ──────────────────────────────────────────────────────

def _make_store_with_docs(docs: list[Document]) -> MagicMock:
    """Create a mock FAISSStore with given documents."""
    store = MagicMock()
    store.total_documents = len(docs)
    store._documents = docs
    store.similarity_search.return_value = [(doc, 0.9) for doc in docs[:3]]
    return store


class TestHybridRetriever:
    def _sample_docs(self) -> list[Document]:
        return [
            Document(
                page_content=f"Document about topic {i}. This is content number {i}.",
                metadata={"chunk_id": f"doc::p0::c{i}", "source": "test.pdf", "page": i},
            )
            for i in range(10)
        ]

    def test_returns_documents(self):
        docs = self._sample_docs()
        store = _make_store_with_docs(docs)
        retriever = HybridRetriever(store)
        results = retriever.retrieve("topic 3", top_k=5)
        assert isinstance(results, list)
        assert all(isinstance(d, Document) for d in results)

    def test_empty_store_returns_empty(self):
        store = MagicMock()
        store.total_documents = 0
        store._documents = []
        retriever = HybridRetriever(store)
        results = retriever.retrieve("anything")
        assert results == []

    def test_results_have_rrf_score(self):
        docs = self._sample_docs()
        store = _make_store_with_docs(docs)
        retriever = HybridRetriever(store)
        results = retriever.retrieve("topic", top_k=5)
        for doc in results:
            assert "rrf_score" in doc.metadata

    def test_top_k_respected(self):
        docs = self._sample_docs()
        store = _make_store_with_docs(docs)
        retriever = HybridRetriever(store)
        results = retriever.retrieve("document", top_k=3)
        assert len(results) <= 3
