"""Tests for results collection and aggregation."""

from __future__ import annotations

import pytest

from embedding_tests.reporting.collector import ModelResult, ResultsCollector


class TestModelResult:
    """Tests for ModelResult dataclass."""

    def test_model_result_has_required_fields(self) -> None:
        result = ModelResult(
            model_name="test-model",
            precision="fp16",
            recall_at_10=0.85,
            mrr=0.7,
            ndcg_at_10=0.75,
            precision_at_10=0.6,
            total_time_seconds=12.5,
        )
        assert result.model_name == "test-model"
        assert result.precision == "fp16"
        assert result.recall_at_10 == 0.85
        assert result.total_time_seconds == 12.5


class TestResultsCollector:
    """Tests for ResultsCollector."""

    def test_collector_aggregates_multiple_results(self) -> None:
        collector = ResultsCollector()
        collector.add(ModelResult("m1", "fp16", 0.8, 0.7, 0.75, 0.6, 10.0))
        collector.add(ModelResult("m2", "fp16", 0.9, 0.8, 0.85, 0.7, 15.0))
        assert len(collector.results) == 2

    def test_collector_filters_by_model(self) -> None:
        collector = ResultsCollector()
        collector.add(ModelResult("m1", "fp16", 0.8, 0.7, 0.75, 0.6, 10.0))
        collector.add(ModelResult("m2", "fp16", 0.9, 0.8, 0.85, 0.7, 15.0))
        filtered = collector.filter_by_model("m1")
        assert len(filtered) == 1
        assert filtered[0].model_name == "m1"

    def test_collector_filters_by_precision(self) -> None:
        collector = ResultsCollector()
        collector.add(ModelResult("m1", "fp16", 0.8, 0.7, 0.75, 0.6, 10.0))
        collector.add(ModelResult("m1", "int8", 0.75, 0.65, 0.7, 0.55, 8.0))
        filtered = collector.filter_by_precision("int8")
        assert len(filtered) == 1
        assert filtered[0].precision == "int8"

    def test_empty_collector_returns_empty_list(self) -> None:
        collector = ResultsCollector()
        assert collector.results == []
        assert collector.filter_by_model("nonexistent") == []

    def test_model_result_with_error(self) -> None:
        result = ModelResult(model_name="test-model", precision="fp16", recall_at_10=0.0, mrr=0.0, ndcg_at_10=0.0, precision_at_10=0.0, total_time_seconds=0.0, error="Model failed to load")
        assert result.error == "Model failed to load"
