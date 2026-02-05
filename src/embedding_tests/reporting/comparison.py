"""Cross-model comparison tables."""

from __future__ import annotations

from typing import Any

from embedding_tests.reporting.collector import ModelResult


_VALID_METRICS = {"recall_at_10", "mrr", "ndcg_at_10", "precision_at_10", "time_seconds"}

_PRECISION_RANK = {"fp16": 0, "int8": 1, "int4": 2, "gptq_int4": 3, "awq_int4": 4}


def cross_model_comparison(
    results: list[ModelResult],
    metric: str = "recall_at_10",
) -> list[dict[str, Any]]:
    """Create a cross-model comparison table ranked by a metric."""
    if metric not in _VALID_METRICS:
        raise ValueError(f"Unknown metric: {metric!r}. Valid: {sorted(_VALID_METRICS)}")
    rows = []
    for r in results:
        row = {
            "model_name": r.model_name,
            "precision": r.precision,
            "recall_at_10": r.recall_at_10,
            "mrr": r.mrr,
            "ndcg_at_10": r.ndcg_at_10,
            "precision_at_10": r.precision_at_10,
            "time_seconds": r.total_time_seconds,
        }
        rows.append(row)

    ascending_metrics = {"time_seconds"}
    rows.sort(key=lambda x: x[metric], reverse=(metric not in ascending_metrics))
    return rows


def precision_impact_table(
    results: list[ModelResult],
    model_name: str,
) -> list[dict[str, Any]]:
    """Show precision impact for a single model."""
    filtered = [r for r in results if r.model_name == model_name]
    rows = []
    for r in filtered:
        rows.append({
            "precision": r.precision,
            "recall_at_10": r.recall_at_10,
            "mrr": r.mrr,
            "ndcg_at_10": r.ndcg_at_10,
            "precision_at_10": r.precision_at_10,
            "time_seconds": r.total_time_seconds,
        })
    rows.sort(key=lambda x: _PRECISION_RANK.get(x["precision"], 99))
    return rows
