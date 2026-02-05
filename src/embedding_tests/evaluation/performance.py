"""Performance metrics: latency and throughput measurement."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


def compute_latency_stats(samples: list[float]) -> dict[str, float]:
    """Compute latency statistics from timing samples.

    Args:
        samples: List of latency measurements in milliseconds.

    Returns:
        Dictionary with mean, min, max, p50, p95, p99 statistics.

    Raises:
        ValueError: If samples list is empty.
    """
    if not samples:
        raise ValueError("Cannot compute latency stats from empty samples")

    sorted_samples = sorted(samples)
    n = len(sorted_samples)

    def percentile(p: float) -> float:
        """Compute percentile using linear interpolation."""
        if n == 1:
            return sorted_samples[0]
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= n:
            return sorted_samples[-1]
        return sorted_samples[f] + (k - f) * (sorted_samples[c] - sorted_samples[f])

    return {
        "mean_ms": sum(samples) / n,
        "min_ms": sorted_samples[0],
        "max_ms": sorted_samples[-1],
        "p50_ms": percentile(50),
        "p95_ms": percentile(95),
        "p99_ms": percentile(99),
    }


def compute_throughput(num_items: int, total_time_seconds: float) -> float:
    """Compute throughput in items per second.

    Args:
        num_items: Number of items processed.
        total_time_seconds: Total processing time in seconds.

    Returns:
        Items per second. Returns infinity for zero time, 0 for zero items.
    """
    if num_items == 0:
        return 0.0
    if total_time_seconds == 0.0:
        return float("inf")
    return num_items / total_time_seconds


def compute_latency_per_item(num_items: int, total_time_ms: float) -> float:
    """Compute average latency per item.

    Args:
        num_items: Number of items processed.
        total_time_ms: Total processing time in milliseconds.

    Returns:
        Milliseconds per item. Returns 0 for zero items.
    """
    if num_items == 0:
        return 0.0
    return total_time_ms / num_items


@dataclass
class PerformanceResult:
    """Result of a performance measurement."""

    operation: str
    total_time_seconds: float
    num_items: int
    throughput_items_per_second: float
    latency_stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "total_time_seconds": self.total_time_seconds,
            "num_items": self.num_items,
            "throughput_items_per_second": self.throughput_items_per_second,
            "latency_stats": self.latency_stats,
        }


class PerformanceTracker:
    """Track performance metrics across multiple operations."""

    def __init__(self) -> None:
        """Initialize the tracker."""
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._batch_sizes: dict[str, list[int]] = defaultdict(list)

    @contextmanager
    def track(
        self, operation: str, batch_size: int = 1
    ) -> Generator[None, None, None]:
        """Context manager to track timing for an operation.

        Args:
            operation: Name of the operation being tracked.
            batch_size: Number of items in this batch.

        Yields:
            None - timing is recorded when context exits.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._timings[operation].append(elapsed)
            self._batch_sizes[operation].append(batch_size)

    def get_results(self) -> dict[str, dict[str, Any]]:
        """Get performance results for all tracked operations.

        Returns:
            Dictionary mapping operation name to performance metrics.
        """
        results: dict[str, dict[str, Any]] = {}

        for operation in self._timings:
            timings = self._timings[operation]
            batch_sizes = self._batch_sizes[operation]

            total_time = sum(timings)
            num_items = sum(batch_sizes)

            throughput = compute_throughput(num_items, total_time)

            # Convert to milliseconds for per-call latency stats.
            # Note: these represent latency per track() call, not per-item.
            latency_ms = [t * 1000 for t in timings]
            latency_stats = compute_latency_stats(latency_ms)

            results[operation] = {
                "total_time_seconds": total_time,
                "num_items": num_items,
                "throughput_items_per_second": throughput,
                "latency_stats": latency_stats,
            }

        return results

    def reset(self) -> None:
        """Clear all accumulated timing data."""
        self._timings.clear()
        self._batch_sizes.clear()
