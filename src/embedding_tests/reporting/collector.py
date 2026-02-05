"""Results collection and aggregation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResult:
    """Result for a single model/precision combination."""

    model_name: str
    precision: str
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    precision_at_10: float
    total_time_seconds: float
    error: str | None = None


class ResultsCollector:
    """Collects and filters experiment results."""

    def __init__(self) -> None:
        self._results: list[ModelResult] = []

    @property
    def results(self) -> list[ModelResult]:
        return list(self._results)

    def add(self, result: ModelResult) -> None:
        self._results.append(result)

    def filter_by_model(self, model_name: str) -> list[ModelResult]:
        return [r for r in self._results if r.model_name == model_name]

    def filter_by_precision(self, precision: str) -> list[ModelResult]:
        return [r for r in self._results if r.precision == precision]
