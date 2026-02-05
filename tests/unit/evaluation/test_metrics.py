"""Tests for retrieval quality metrics."""

from __future__ import annotations

import pytest

from embedding_tests.evaluation.metrics import (
    recall_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
)


class TestRecallAtK:
    """Tests for Recall@k metric."""

    def test_recall_at_1_perfect_retrieval(self) -> None:
        retrieved = ["d1"]
        relevant = {"d1"}
        assert recall_at_k(retrieved, relevant, k=1) == 1.0

    def test_recall_at_1_no_relevant(self) -> None:
        retrieved = ["d2"]
        relevant = {"d1"}
        assert recall_at_k(retrieved, relevant, k=1) == 0.0

    def test_recall_at_k_partial(self) -> None:
        retrieved = ["d1", "d3", "d2"]
        relevant = {"d1", "d2"}
        # top-2: d1, d3 -> 1 of 2 relevant = 0.5
        assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)

    def test_recall_at_k_all_relevant(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2"}
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_recall_at_k_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k(["d1"], {"d1"}, k=0)


class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_mrr_perfect(self) -> None:
        # Relevant doc always at rank 1
        queries = [
            (["d1", "d2"], {"d1"}),
            (["d3", "d4"], {"d3"}),
        ]
        score = mrr(queries)
        assert score == pytest.approx(1.0)

    def test_mrr_at_rank_3(self) -> None:
        queries = [
            (["d2", "d3", "d1"], {"d1"}),
        ]
        score = mrr(queries)
        assert score == pytest.approx(1 / 3)

    def test_mrr_no_relevant(self) -> None:
        queries = [
            (["d2", "d3"], {"d1"}),
        ]
        score = mrr(queries)
        assert score == 0.0


class TestNDCG:
    """Tests for NDCG@k."""

    def test_ndcg_perfect_ranking(self) -> None:
        # Single relevant doc at position 1
        retrieved = ["d1", "d2", "d3"]
        relevance = {"d1": 1.0}
        score = ndcg_at_k(retrieved, relevance, k=3)
        assert score == pytest.approx(1.0)

    def test_ndcg_worst_ranking(self) -> None:
        # Relevant doc not in retrieved
        retrieved = ["d2", "d3"]
        relevance = {"d1": 1.0}
        score = ndcg_at_k(retrieved, relevance, k=2)
        assert score == 0.0

    def test_ndcg_graded_relevance(self) -> None:
        retrieved = ["d2", "d1", "d3"]
        relevance = {"d1": 3.0, "d2": 1.0, "d3": 0.0}
        score = ndcg_at_k(retrieved, relevance, k=3)
        # DCG  = 1.0/log2(2) + 3.0/log2(3) + 0.0/log2(4) ≈ 2.893
        # IDCG = 3.0/log2(2) + 1.0/log2(3) + 0.0/log2(4) ≈ 3.631
        # NDCG ≈ 0.797
        assert score == pytest.approx(0.797, abs=0.001)

    def test_ndcg_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            ndcg_at_k(["d1"], {"d1": 1.0}, k=0)


class TestPrecisionAtK:
    """Tests for Precision@k."""

    def test_precision_at_k_all_relevant(self) -> None:
        retrieved = ["d1", "d2"]
        relevant = {"d1", "d2", "d3"}
        assert precision_at_k(retrieved, relevant, k=2) == 1.0

    def test_precision_at_k_none_relevant(self) -> None:
        retrieved = ["d4", "d5"]
        relevant = {"d1", "d2"}
        assert precision_at_k(retrieved, relevant, k=2) == 0.0

    def test_precision_at_k_half(self) -> None:
        retrieved = ["d1", "d4"]
        relevant = {"d1", "d2"}
        assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)

    def test_precision_at_k_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            precision_at_k(["d1"], {"d1"}, k=0)
