"""Results collection and aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


@dataclass
class ComprehensiveResult:
    """Comprehensive result with all metrics for a model/precision combination."""

    model_name: str
    precision: str
    status: str

    # Aggregate metrics (mean values for quick access)
    mrr: float = 0.0
    map: float = 0.0

    # Per-k metrics with statistics (mean, std, min, max, median, p25, p75)
    recall_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    precision_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    ndcg_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    f1_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    success_rates: dict[int, float] = field(default_factory=dict)
    r_precision_stats: dict[str, float] = field(default_factory=dict)

    # Performance metrics
    total_time_seconds: float = 0.0
    embedding_time_seconds: float = 0.0
    num_corpus_chunks: int = 0
    num_queries: int = 0
    queries_per_second: float = 0.0

    # Per-query results (optional, for detailed analysis)
    per_query_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    error: str | None = None

    @classmethod
    def from_experiment_result(cls, result: dict[str, Any]) -> "ComprehensiveResult":
        """Create from experiment runner output."""
        aggregate = result.get("aggregate", {})
        performance = result.get("performance", {})

        # Extract per-k statistics
        k_values = [1, 3, 5, 10, 20]
        recall_stats = {}
        precision_stats = {}
        ndcg_stats = {}
        f1_stats = {}
        success_rates = {}

        for k in k_values:
            if f"recall_at_{k}" in aggregate:
                recall_stats[k] = aggregate[f"recall_at_{k}"]
            if f"precision_at_{k}" in aggregate:
                precision_stats[k] = aggregate[f"precision_at_{k}"]
            if f"ndcg_at_{k}" in aggregate:
                ndcg_stats[k] = aggregate[f"ndcg_at_{k}"]
            if f"f1_at_{k}" in aggregate:
                f1_stats[k] = aggregate[f"f1_at_{k}"]
            if f"success_at_{k}" in aggregate:
                success_rates[k] = aggregate[f"success_at_{k}"]

        return cls(
            model_name=result.get("model", ""),
            precision=result.get("precision", ""),
            status=result.get("status", "unknown"),
            mrr=result.get("mrr", aggregate.get("mrr", 0.0)),
            map=result.get("map", aggregate.get("map", 0.0)),
            recall_stats=recall_stats,
            precision_stats=precision_stats,
            ndcg_stats=ndcg_stats,
            f1_stats=f1_stats,
            success_rates=success_rates,
            r_precision_stats=aggregate.get("r_precision", {}),
            total_time_seconds=performance.get(
                "total_time_seconds", result.get("total_time", 0.0)
            ),
            embedding_time_seconds=performance.get("embedding_time_seconds", 0.0),
            num_corpus_chunks=performance.get("num_corpus_chunks", 0),
            num_queries=performance.get("num_queries", 0),
            queries_per_second=performance.get("queries_per_second", 0.0),
            per_query_results=result.get("results", {}),
            error=result.get("error"),
        )

    def to_legacy_result(self) -> ModelResult:
        """Convert to legacy ModelResult for backward compatibility."""
        return ModelResult(
            model_name=self.model_name,
            precision=self.precision,
            recall_at_10=self.recall_stats.get(10, {}).get("mean", 0.0),
            mrr=self.mrr,
            ndcg_at_10=self.ndcg_stats.get(10, {}).get("mean", 0.0),
            precision_at_10=self.precision_stats.get(10, {}).get("mean", 0.0),
            total_time_seconds=self.total_time_seconds,
            error=self.error,
        )


class ResultsCollector:
    """Collects and filters experiment results."""

    def __init__(self) -> None:
        self._results: list[ModelResult] = []
        self._comprehensive_results: list[ComprehensiveResult] = []

    @property
    def results(self) -> list[ModelResult]:
        return list(self._results)

    @property
    def comprehensive_results(self) -> list[ComprehensiveResult]:
        return list(self._comprehensive_results)

    def add(self, result: ModelResult) -> None:
        self._results.append(result)

    def add_comprehensive(self, result: ComprehensiveResult) -> None:
        self._comprehensive_results.append(result)
        # Also add legacy version for backward compatibility
        self._results.append(result.to_legacy_result())

    def filter_by_model(self, model_name: str) -> list[ModelResult]:
        return [r for r in self._results if r.model_name == model_name]

    def filter_by_precision(self, precision: str) -> list[ModelResult]:
        return [r for r in self._results if r.precision == precision]

    def filter_comprehensive_by_model(
        self, model_name: str
    ) -> list[ComprehensiveResult]:
        return [r for r in self._comprehensive_results if r.model_name == model_name]
