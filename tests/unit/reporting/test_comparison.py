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
