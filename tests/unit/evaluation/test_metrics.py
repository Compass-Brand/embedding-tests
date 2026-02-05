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
