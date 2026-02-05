"""Tests for RAG evaluator."""

from __future__ import annotations

import pytest

from embedding_tests.evaluation.rag_evaluator import (
    compute_context_recall,
    compute_context_precision,
    aggregate_scores,
)


class TestContextRecall:
    """Tests for context recall computation."""

    def test_context_recall_perfect(self) -> None:
        retrieved = ["d1", "d2"]
        relevant = ["d1", "d2"]
        assert compute_context_recall(retrieved, relevant) == 1.0

    def test_context_recall_none(self) -> None:
        retrieved = ["d3", "d4"]
        relevant = ["d1", "d2"]
        assert compute_context_recall(retrieved, relevant) == 0.0

    def test_context_recall_partial(self) -> None:
        retrieved = ["d1", "d3"]
        relevant = ["d1", "d2"]
        assert compute_context_recall(retrieved, relevant) == pytest.approx(0.5)

    def test_context_recall_empty_relevant(self) -> None:
        retrieved = ["d1", "d2"]
        relevant: list[str] = []
        assert compute_context_recall(retrieved, relevant) == 0.0

    def test_context_recall_both_empty(self) -> None:
        assert compute_context_recall([], []) == 0.0

    def test_context_recall_with_duplicates(self) -> None:
        """Verify duplicates in retrieved are deduplicated via set conversion."""
        retrieved = ["d1", "d1", "d2"]
        relevant = ["d1", "d2"]
        # set(retrieved) = {"d1", "d2"}, intersection = {"d1", "d2"} -> 2/2 = 1.0
        assert compute_context_recall(retrieved, relevant) == 1.0


class TestContextPrecision:
    """Tests for context precision computation."""

    def test_context_precision_perfect(self) -> None:
        retrieved = ["d1", "d2"]
        relevant = ["d1", "d2"]
        assert compute_context_precision(retrieved, relevant) == 1.0

    def test_context_precision_half(self) -> None:
        retrieved = ["d1", "d3"]
        relevant = ["d1", "d2"]
        assert compute_context_precision(retrieved, relevant) == pytest.approx(0.5)

    def test_context_precision_empty_retrieved(self) -> None:
        retrieved: list[str] = []
        relevant = ["d1", "d2"]
        assert compute_context_precision(retrieved, relevant) == 0.0

    def test_context_precision_none_relevant(self) -> None:
        retrieved = ["d3", "d4"]
        relevant = ["d1", "d2"]
        assert compute_context_precision(retrieved, relevant) == 0.0


class TestAggregateScores:
    """Tests for score aggregation."""

    def test_aggregate_across_queries(self) -> None:
        scores = [0.5, 1.0, 0.0]
        assert aggregate_scores(scores) == pytest.approx(0.5)

    def test_aggregate_empty(self) -> None:
        assert aggregate_scores([]) == 0.0
