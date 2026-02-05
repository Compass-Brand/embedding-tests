"""Tests for performance metrics (latency, throughput)."""

from __future__ import annotations

import pytest


class TestLatencyStats:
    """Tests for latency statistics computation."""

    def test_latency_stats_from_samples(self) -> None:
        """Should compute latency statistics from timing samples."""
        from embedding_tests.evaluation.performance import compute_latency_stats

        samples = [10.0, 20.0, 30.0, 40.0, 50.0]  # milliseconds
        stats = compute_latency_stats(samples)

        assert stats["mean_ms"] == pytest.approx(30.0)
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 50.0

    def test_latency_stats_percentiles(self) -> None:
        """Should compute p50, p95, p99 percentiles."""
        from embedding_tests.evaluation.performance import compute_latency_stats

        # 100 samples: 1 to 100
        samples = list(range(1, 101))
        stats = compute_latency_stats(samples)

        # p50 (median) ≈ 50
        assert stats["p50_ms"] == pytest.approx(50.5, abs=1)
        # p95 ≈ 95
        assert stats["p95_ms"] == pytest.approx(95.05, abs=1)
        # p99 ≈ 99
        assert stats["p99_ms"] == pytest.approx(99.01, abs=1)

    def test_latency_stats_empty_raises(self) -> None:
        """Should raise for empty samples."""
        from embedding_tests.evaluation.performance import compute_latency_stats

        with pytest.raises(ValueError, match="empty"):
            compute_latency_stats([])

    def test_latency_stats_single_sample(self) -> None:
        """Should handle single sample gracefully."""
        from embedding_tests.evaluation.performance import compute_latency_stats

        stats = compute_latency_stats([100.0])

        assert stats["mean_ms"] == 100.0
        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 100.0
        assert stats["p50_ms"] == 100.0


class TestThroughputMetrics:
    """Tests for throughput computation."""

    def test_compute_throughput(self) -> None:
        """Should compute items per second."""
        from embedding_tests.evaluation.performance import compute_throughput

        # 100 items in 2 seconds = 50 items/sec
        throughput = compute_throughput(num_items=100, total_time_seconds=2.0)
        assert throughput == pytest.approx(50.0)

    def test_compute_throughput_zero_time_returns_inf(self) -> None:
        """Should return infinity for zero time (instantaneous)."""
        from embedding_tests.evaluation.performance import compute_throughput

        throughput = compute_throughput(num_items=100, total_time_seconds=0.0)
        assert throughput == float("inf")

    def test_compute_throughput_zero_items(self) -> None:
        """Should return 0 for zero items."""
        from embedding_tests.evaluation.performance import compute_throughput

        throughput = compute_throughput(num_items=0, total_time_seconds=1.0)
        assert throughput == 0.0


class TestPerformanceResult:
    """Tests for performance result dataclass."""

    def test_performance_result_creation(self) -> None:
        """Should create performance result with all fields."""
        from embedding_tests.evaluation.performance import PerformanceResult

        result = PerformanceResult(
            operation="encode",
            total_time_seconds=2.5,
            num_items=100,
            throughput_items_per_second=40.0,
            latency_stats={
                "mean_ms": 25.0,
                "min_ms": 10.0,
                "max_ms": 50.0,
                "p50_ms": 24.0,
                "p95_ms": 45.0,
                "p99_ms": 48.0,
            },
        )

        assert result.operation == "encode"
        assert result.total_time_seconds == 2.5
        assert result.throughput_items_per_second == 40.0

    def test_performance_result_to_dict(self) -> None:
        """Should convert to dictionary for serialization."""
        from embedding_tests.evaluation.performance import PerformanceResult

        result = PerformanceResult(
            operation="rerank",
            total_time_seconds=1.0,
            num_items=50,
            throughput_items_per_second=50.0,
            latency_stats={"mean_ms": 20.0},
        )

        d = result.to_dict()

        assert d["operation"] == "rerank"
        assert d["throughput_items_per_second"] == 50.0
        assert "latency_stats" in d


class TestPerformanceTracker:
    """Tests for performance tracking during operations."""

    def test_tracker_records_timing(self) -> None:
        """Tracker should record timing for operations."""
        from embedding_tests.evaluation.performance import PerformanceTracker
        import time

        tracker = PerformanceTracker()

        with tracker.track("encode"):
            time.sleep(0.01)  # 10ms

        results = tracker.get_results()
        assert "encode" in results
        assert results["encode"]["total_time_seconds"] >= 0.01

    def test_tracker_accumulates_batches(self) -> None:
        """Tracker should accumulate timing across batches."""
        from embedding_tests.evaluation.performance import PerformanceTracker
        import time

        tracker = PerformanceTracker()

        for _ in range(3):
            with tracker.track("encode", batch_size=10):
                time.sleep(0.005)  # 5ms per batch

        results = tracker.get_results()
        assert results["encode"]["num_items"] == 30  # 3 batches * 10
        assert results["encode"]["total_time_seconds"] >= 0.015  # At least 15ms

    def test_tracker_computes_throughput(self) -> None:
        """Tracker should compute throughput from accumulated data."""
        from embedding_tests.evaluation.performance import PerformanceTracker

        tracker = PerformanceTracker()

        # Manually add timing data for deterministic test
        tracker._timings["encode"] = [0.1, 0.1]  # 100ms each
        tracker._batch_sizes["encode"] = [50, 50]

        results = tracker.get_results()
        # 100 items in 0.2 seconds = 500 items/sec
        assert results["encode"]["throughput_items_per_second"] == pytest.approx(500.0)

    def test_tracker_reset_clears_data(self) -> None:
        """Tracker reset should clear all accumulated data."""
        from embedding_tests.evaluation.performance import PerformanceTracker
        import time

        tracker = PerformanceTracker()

        with tracker.track("encode", batch_size=10):
            time.sleep(0.001)

        tracker.reset()
        results = tracker.get_results()

        assert results == {}


class TestLatencyPerItem:
    """Tests for per-item latency computation."""

    def test_latency_per_item(self) -> None:
        """Should compute average latency per item."""
        from embedding_tests.evaluation.performance import compute_latency_per_item

        # 100 items in 500ms = 5ms per item
        latency = compute_latency_per_item(num_items=100, total_time_ms=500.0)
        assert latency == pytest.approx(5.0)

    def test_latency_per_item_zero_items(self) -> None:
        """Should return 0 for zero items."""
        from embedding_tests.evaluation.performance import compute_latency_per_item

        latency = compute_latency_per_item(num_items=0, total_time_ms=500.0)
        assert latency == 0.0
