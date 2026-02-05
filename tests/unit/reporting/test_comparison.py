"""Tests for cross-model comparison."""

from __future__ import annotations

import pytest

from embedding_tests.reporting.collector import ModelResult
from embedding_tests.reporting.comparison import (
    cross_model_comparison,
    precision_impact_table,
)


class TestCrossModelComparison:
    """Tests for cross-model comparison tables."""

    def test_cross_model_comparison_table(self) -> None:
        results = [
            ModelResult("m1", "fp16", 0.8, 0.7, 0.75, 0.6, 10.0),
            ModelResult("m2", "fp16", 0.9, 0.8, 0.85, 0.7, 15.0),
        ]
        table = cross_model_comparison(results, metric="recall_at_10")
        assert len(table) == 2
        # Should be ranked by metric descending
        assert table[0]["model_name"] == "m2"
        assert table[0]["recall_at_10"] == 0.9

    def test_cross_model_comparison_invalid_metric_raises(self) -> None:
        results = [ModelResult("m1", "fp16", 0.8, 0.7, 0.75, 0.6, 10.0)]
        with pytest.raises(ValueError, match="Unknown metric"):
            cross_model_comparison(results, metric="invalid_metric")

    def test_precision_impact_table(self) -> None:
        results = [
            ModelResult("m1", "fp16", 0.8, 0.7, 0.75, 0.6, 10.0),
            ModelResult("m1", "int8", 0.75, 0.65, 0.7, 0.55, 8.0),
        ]
        table = precision_impact_table(results, model_name="m1")
        assert len(table) == 2
        # Both precisions for same model
        precisions = {row["precision"] for row in table}
        assert precisions == {"fp16", "int8"}
        # Sorted by precision rank: fp16 first, int8 second
        assert table[0]["precision"] == "fp16"
        assert table[1]["precision"] == "int8"

    def test_precision_impact_table_nonexistent_model(self) -> None:
        results = [ModelResult("m1", "fp16", 0.8, 0.7, 0.75, 0.6, 10.0)]
        table = precision_impact_table(results, model_name="nonexistent")
        assert table == []

    def test_cross_model_comparison_excludes_error_results(self) -> None:
        results = [
            ModelResult("m1", "fp16", 0.8, 0.7, 0.75, 0.6, 10.0),
            ModelResult("m2", "fp16", 0.0, 0.0, 0.0, 0.0, 0.0, error="Failed"),
        ]
        table = cross_model_comparison(results, metric="recall_at_10")
        assert len(table) == 1
        assert table[0]["model_name"] == "m1"

    def test_precision_impact_table_excludes_error_results(self) -> None:
        results = [
            ModelResult("m1", "fp16", 0.8, 0.7, 0.75, 0.6, 10.0),
            ModelResult("m1", "int8", 0.0, 0.0, 0.0, 0.0, 0.0, error="OOM"),
        ]
        table = precision_impact_table(results, model_name="m1")
        assert len(table) == 1
        assert table[0]["precision"] == "fp16"

    def test_cross_model_comparison_time_seconds_sorts_ascending(self) -> None:
        results = [
            ModelResult("fast", "fp16", 0.8, 0.7, 0.75, 0.6, 5.0),
            ModelResult("slow", "fp16", 0.9, 0.8, 0.85, 0.7, 20.0),
            ModelResult("mid", "fp16", 0.85, 0.75, 0.8, 0.65, 10.0),
        ]
        table = cross_model_comparison(results, metric="time_seconds")
        # time_seconds should sort ascending (fastest first)
        times = [row["time_seconds"] for row in table]
        assert times == sorted(times)
        assert table[0]["model_name"] == "fast"
        assert table[2]["model_name"] == "slow"
