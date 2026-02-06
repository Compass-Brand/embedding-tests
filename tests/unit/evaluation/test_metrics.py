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

    def test_mrr_empty_queries(self) -> None:
        assert mrr([]) == 0.0


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

    def test_precision_at_k_exceeds_retrieved_length(self) -> None:
        retrieved = ["d1"]
        relevant = {"d1", "d2"}
        assert precision_at_k(retrieved, relevant, k=5) == 1.0

    def test_precision_at_k_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            precision_at_k(["d1"], {"d1"}, k=0)


class TestMeanAveragePrecision:
    """Tests for Mean Average Precision (MAP)."""

    def test_map_perfect_ranking(self) -> None:
        """MAP should be 1.0 when all relevant docs are at top positions."""
        from embedding_tests.evaluation.metrics import mean_average_precision

        # Query with 2 relevant docs at positions 1 and 2
        queries = [
            (["d1", "d2", "d3"], {"d1", "d2"}),
        ]
        # AP = (1/1 + 2/2) / 2 = 1.0
        assert mean_average_precision(queries) == pytest.approx(1.0)

    def test_map_worst_ranking(self) -> None:
        """MAP should be low when relevant docs are at bottom."""
        from embedding_tests.evaluation.metrics import mean_average_precision

        # Relevant docs at positions 3 and 4
        queries = [
            (["d3", "d4", "d1", "d2"], {"d1", "d2"}),
        ]
        # AP = (1/3 + 2/4) / 2 = (0.333 + 0.5) / 2 = 0.4167
        assert mean_average_precision(queries) == pytest.approx(0.4167, abs=0.001)

    def test_map_multiple_queries(self) -> None:
        """MAP should average across multiple queries."""
        from embedding_tests.evaluation.metrics import mean_average_precision

        queries = [
            (["d1", "d2"], {"d1"}),  # AP = 1.0
            (["d2", "d1"], {"d1"}),  # AP = 1/2 = 0.5
        ]
        # MAP = (1.0 + 0.5) / 2 = 0.75
        assert mean_average_precision(queries) == pytest.approx(0.75)

    def test_map_no_relevant_found(self) -> None:
        """MAP should be 0 when no relevant docs in retrieved."""
        from embedding_tests.evaluation.metrics import mean_average_precision

        queries = [
            (["d3", "d4"], {"d1", "d2"}),
        ]
        assert mean_average_precision(queries) == 0.0

    def test_map_empty_queries(self) -> None:
        """MAP should be 0 for empty query list."""
        from embedding_tests.evaluation.metrics import mean_average_precision

        assert mean_average_precision([]) == 0.0

    def test_map_excludes_empty_relevant_sets(self) -> None:
        """MAP should exclude queries with empty relevant sets from average."""
        from embedding_tests.evaluation.metrics import mean_average_precision

        queries = [
            (["d1", "d2"], {"d1"}),  # AP = 1.0
            (["d3", "d4"], set()),  # Excluded - no ground truth
        ]
        # Only one valid query, so MAP = 1.0 (not 0.5)
        assert mean_average_precision(queries) == pytest.approx(1.0)

    def test_map_all_empty_relevant_sets(self) -> None:
        """MAP should be 0 when all queries have empty relevant sets."""
        from embedding_tests.evaluation.metrics import mean_average_precision

        queries = [
            (["d1", "d2"], set()),
            (["d3", "d4"], set()),
        ]
        assert mean_average_precision(queries) == 0.0


class TestSuccessAtK:
    """Tests for Success@k metric."""

    def test_success_at_k_found(self) -> None:
        """Success@k should be 1.0 when any relevant doc in top-k."""
        from embedding_tests.evaluation.metrics import success_at_k

        retrieved = ["d2", "d1", "d3"]
        relevant = {"d1"}
        assert success_at_k(retrieved, relevant, k=2) == 1.0

    def test_success_at_k_not_found(self) -> None:
        """Success@k should be 0.0 when no relevant doc in top-k."""
        from embedding_tests.evaluation.metrics import success_at_k

        retrieved = ["d2", "d3", "d1"]
        relevant = {"d1"}
        assert success_at_k(retrieved, relevant, k=2) == 0.0

    def test_success_at_k_at_boundary(self) -> None:
        """Success@k should find doc at exactly position k."""
        from embedding_tests.evaluation.metrics import success_at_k

        retrieved = ["d2", "d3", "d1"]
        relevant = {"d1"}
        assert success_at_k(retrieved, relevant, k=3) == 1.0

    def test_success_at_k_multiple_relevant(self) -> None:
        """Success@k is still 1.0 with multiple relevant docs (binary metric)."""
        from embedding_tests.evaluation.metrics import success_at_k

        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2"}
        assert success_at_k(retrieved, relevant, k=1) == 1.0

    def test_success_at_k_empty_relevant(self) -> None:
        """Success@k should be 0.0 when no relevant docs specified."""
        from embedding_tests.evaluation.metrics import success_at_k

        retrieved = ["d1", "d2"]
        relevant: set[str] = set()
        assert success_at_k(retrieved, relevant, k=2) == 0.0

    def test_success_at_k_invalid_k_raises(self) -> None:
        """Success@k should raise for invalid k."""
        from embedding_tests.evaluation.metrics import success_at_k

        with pytest.raises(ValueError, match="k must be positive"):
            success_at_k(["d1"], {"d1"}, k=0)


class TestRecallAtMultipleK:
    """Tests for multi-k recall computation."""

    def test_recall_at_multiple_k_all_values(self) -> None:
        """Should compute recall at multiple k values."""
        from embedding_tests.evaluation.metrics import recall_at_multiple_k

        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = {"d1", "d3", "d5"}
        k_values = [1, 3, 5]

        result = recall_at_multiple_k(retrieved, relevant, k_values)

        assert result[1] == pytest.approx(1 / 3)  # d1 found
        assert result[3] == pytest.approx(2 / 3)  # d1, d3 found
        assert result[5] == pytest.approx(1.0)  # all found

    def test_recall_at_multiple_k_default_values(self) -> None:
        """Should use default k values (1, 3, 5, 10, 20)."""
        from embedding_tests.evaluation.metrics import recall_at_multiple_k

        retrieved = ["d1"] * 20
        relevant = {"d1"}

        result = recall_at_multiple_k(retrieved, relevant)

        assert 1 in result
        assert 3 in result
        assert 5 in result
        assert 10 in result
        assert 20 in result

    def test_recall_at_multiple_k_empty_k_values(self) -> None:
        """Should raise for empty k_values list."""
        from embedding_tests.evaluation.metrics import recall_at_multiple_k

        with pytest.raises(ValueError, match="k_values must not be empty"):
            recall_at_multiple_k(["d1"], {"d1"}, [])


class TestPrecisionAtMultipleK:
    """Tests for multi-k precision computation."""

    def test_precision_at_multiple_k_all_values(self) -> None:
        """Should compute precision at multiple k values."""
        from embedding_tests.evaluation.metrics import precision_at_multiple_k

        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = {"d1", "d3", "d5"}
        k_values = [1, 3, 5]

        result = precision_at_multiple_k(retrieved, relevant, k_values)

        assert result[1] == pytest.approx(1.0)  # d1 is relevant, 1/1
        assert result[3] == pytest.approx(2 / 3)  # d1, d3 relevant out of 3
        assert result[5] == pytest.approx(3 / 5)  # all 3 relevant out of 5

    def test_precision_at_multiple_k_default_values(self) -> None:
        """Should use default k values (1, 3, 5, 10, 20)."""
        from embedding_tests.evaluation.metrics import precision_at_multiple_k

        retrieved = ["d1"] * 20
        relevant = {"d1"}

        result = precision_at_multiple_k(retrieved, relevant)

        assert 1 in result
        assert 3 in result
        assert 5 in result
        assert 10 in result
        assert 20 in result


class TestNdcgAtMultipleK:
    """Tests for multi-k NDCG computation."""

    def test_ndcg_at_multiple_k_all_values(self) -> None:
        """Should compute NDCG at multiple k values."""
        from embedding_tests.evaluation.metrics import ndcg_at_multiple_k

        retrieved = ["d1", "d2", "d3"]
        relevance = {"d1": 1.0, "d3": 1.0}
        k_values = [1, 2, 3]

        result = ndcg_at_multiple_k(retrieved, relevance, k_values)

        assert result[1] == pytest.approx(1.0)  # d1 relevant at pos 1
        assert 2 in result
        assert 3 in result

    def test_ndcg_at_multiple_k_default_values(self) -> None:
        """Should use default k values (1, 3, 5, 10, 20)."""
        from embedding_tests.evaluation.metrics import ndcg_at_multiple_k

        retrieved = ["d1"] * 20
        relevance = {"d1": 1.0}

        result = ndcg_at_multiple_k(retrieved, relevance)

        assert 1 in result
        assert 3 in result
        assert 5 in result
        assert 10 in result
        assert 20 in result


class TestF1AtK:
    """Tests for F1@k metric."""

    def test_f1_at_k_perfect(self) -> None:
        """F1@k should be 1.0 when precision and recall are both 1.0."""
        from embedding_tests.evaluation.metrics import f1_at_k

        retrieved = ["d1", "d2"]
        relevant = {"d1", "d2"}
        assert f1_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)

    def test_f1_at_k_zero(self) -> None:
        """F1@k should be 0.0 when no relevant docs found."""
        from embedding_tests.evaluation.metrics import f1_at_k

        retrieved = ["d3", "d4"]
        relevant = {"d1", "d2"}
        assert f1_at_k(retrieved, relevant, k=2) == 0.0

    def test_f1_at_k_balanced(self) -> None:
        """F1@k should be harmonic mean of P and R."""
        from embedding_tests.evaluation.metrics import f1_at_k

        # P@2 = 1/2 = 0.5, R@2 = 1/2 = 0.5 (1 of 2 relevant found)
        retrieved = ["d1", "d3"]
        relevant = {"d1", "d2"}
        # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert f1_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)

    def test_f1_at_k_unbalanced(self) -> None:
        """F1@k should favor balance between P and R."""
        from embedding_tests.evaluation.metrics import f1_at_k

        # P@1 = 1.0, R@1 = 1/3 (1 of 3 relevant found)
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        # F1 = 2 * 1.0 * (1/3) / (1.0 + 1/3) = 0.5
        assert f1_at_k(retrieved, relevant, k=1) == pytest.approx(0.5)

    def test_f1_at_k_invalid_k_raises(self) -> None:
        """F1@k should raise for invalid k."""
        from embedding_tests.evaluation.metrics import f1_at_k

        with pytest.raises(ValueError, match="k must be positive"):
            f1_at_k(["d1"], {"d1"}, k=0)


class TestRPrecision:
    """Tests for R-Precision metric."""

    def test_r_precision_perfect(self) -> None:
        """R-Precision should be 1.0 when top R docs are all relevant."""
        from embedding_tests.evaluation.metrics import r_precision

        retrieved = ["d1", "d2", "d3", "d4"]
        relevant = {"d1", "d2"}  # R = 2
        assert r_precision(retrieved, relevant) == pytest.approx(1.0)

    def test_r_precision_zero(self) -> None:
        """R-Precision should be 0.0 when no relevant docs in top R."""
        from embedding_tests.evaluation.metrics import r_precision

        retrieved = ["d3", "d4", "d1", "d2"]
        relevant = {"d1", "d2"}  # R = 2
        assert r_precision(retrieved, relevant) == 0.0

    def test_r_precision_partial(self) -> None:
        """R-Precision should be partial when some relevant docs in top R."""
        from embedding_tests.evaluation.metrics import r_precision

        retrieved = ["d1", "d3", "d2", "d4"]
        relevant = {"d1", "d2"}  # R = 2, top 2 has only d1
        assert r_precision(retrieved, relevant) == pytest.approx(0.5)

    def test_r_precision_empty_relevant(self) -> None:
        """R-Precision should be 0.0 when no relevant docs specified."""
        from embedding_tests.evaluation.metrics import r_precision

        retrieved = ["d1", "d2"]
        relevant: set[str] = set()
        assert r_precision(retrieved, relevant) == 0.0


class TestMeanSuccessAtK:
    """Tests for Mean Success@k (Hit Rate)."""

    def test_mean_success_at_k_all_hits(self) -> None:
        """Should be 1.0 when all queries have hits in top-k."""
        from embedding_tests.evaluation.metrics import mean_success_at_k

        queries = [
            (["d1", "d2"], {"d1"}),
            (["d3", "d4"], {"d3"}),
        ]
        assert mean_success_at_k(queries, k=2) == pytest.approx(1.0)

    def test_mean_success_at_k_no_hits(self) -> None:
        """Should be 0.0 when no queries have hits in top-k."""
        from embedding_tests.evaluation.metrics import mean_success_at_k

        queries = [
            (["d1", "d2"], {"d5"}),
            (["d3", "d4"], {"d6"}),
        ]
        assert mean_success_at_k(queries, k=2) == 0.0

    def test_mean_success_at_k_partial(self) -> None:
        """Should average hit rate across queries."""
        from embedding_tests.evaluation.metrics import mean_success_at_k

        queries = [
            (["d1", "d2"], {"d1"}),  # hit
            (["d3", "d4"], {"d5"}),  # miss
        ]
        assert mean_success_at_k(queries, k=2) == pytest.approx(0.5)

    def test_mean_success_at_k_empty_queries(self) -> None:
        """Should be 0.0 for empty query list."""
        from embedding_tests.evaluation.metrics import mean_success_at_k

        assert mean_success_at_k([], k=10) == 0.0


class TestComputeAggregateStats:
    """Tests for aggregate statistics computation."""

    def test_compute_aggregate_stats_basic(self) -> None:
        """Should compute correct statistics for a simple list."""
        from embedding_tests.evaluation.metrics import compute_aggregate_stats

        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_aggregate_stats(scores)

        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["median"] == pytest.approx(3.0)

    def test_compute_aggregate_stats_single_value(self) -> None:
        """Should handle single value correctly."""
        from embedding_tests.evaluation.metrics import compute_aggregate_stats

        scores = [0.5]
        stats = compute_aggregate_stats(scores)

        assert stats["mean"] == pytest.approx(0.5)
        assert stats["std"] == pytest.approx(0.0)
        assert stats["min"] == pytest.approx(0.5)
        assert stats["max"] == pytest.approx(0.5)
        assert stats["median"] == pytest.approx(0.5)

    def test_compute_aggregate_stats_empty(self) -> None:
        """Should return zeros for empty list."""
        from embedding_tests.evaluation.metrics import compute_aggregate_stats

        stats = compute_aggregate_stats([])

        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0

    def test_compute_aggregate_stats_std(self) -> None:
        """Should compute correct standard deviation."""
        from embedding_tests.evaluation.metrics import compute_aggregate_stats

        # Sample std for [0, 2] = sqrt(2)
        scores = [0.0, 2.0]
        stats = compute_aggregate_stats(scores)

        assert stats["std"] == pytest.approx(1.414, abs=0.001)
