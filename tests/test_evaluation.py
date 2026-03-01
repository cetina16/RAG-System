"""Tests for evaluation metrics."""
from __future__ import annotations

import pytest
from app.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    hit_rate_at_k,
    compute_all_metrics,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, k=3) == 0.0

    def test_partial(self):
        assert precision_at_k(["a", "b", "c"], {"a", "c"}, k=3) == pytest.approx(2/3)

    def test_k_truncates(self):
        assert precision_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0


class TestReciprocalRank:
    def test_first_hit(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_hit(self):
        assert reciprocal_rank(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_third_hit(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1/3)

    def test_no_hit(self):
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0


class TestMeanReciprocalRank:
    def test_perfect_mrr(self):
        all_r = [["a"], ["b"]]
        all_rel = [{"a"}, {"b"}]
        assert mean_reciprocal_rank(all_r, all_rel) == 1.0

    def test_mrr_average(self):
        all_r = [["a"], ["x", "b"]]
        all_rel = [{"a"}, {"b"}]
        assert mean_reciprocal_rank(all_r, all_rel) == pytest.approx(0.75)

    def test_empty(self):
        assert mean_reciprocal_rank([], []) == 0.0


class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_k_matters(self):
        assert hit_rate_at_k(["x", "a"], {"a"}, k=1) == 0.0
        assert hit_rate_at_k(["x", "a"], {"a"}, k=2) == 1.0


class TestComputeAllMetrics:
    def test_structure(self):
        result = compute_all_metrics(["a", "b", "c"], {"a"}, k=3)
        assert "precision@3" in result
        assert "recall@3"    in result
        assert "mrr"         in result
        assert "hit_rate@3"  in result

    def test_values_in_range(self):
        result = compute_all_metrics(["a", "b"], {"a", "c"}, k=2)
        for v in result.values():
            assert 0.0 <= v <= 1.0
