"""Multi-format result export (JSON, CSV, Markdown)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from embedding_tests.reporting.collector import ComprehensiveResult, ModelResult

_FIELDS = [
    "model_name", "precision", "recall_at_10", "mrr",
    "ndcg_at_10", "precision_at_10", "total_time_seconds", "error",
]

_HEADER_LABELS: dict[str, str] = {
    "model_name": "model_name",
    "precision": "precision",
    "recall_at_10": "recall@10",
    "mrr": "MRR",
    "ndcg_at_10": "NDCG@10",
    "precision_at_10": "P@10",
    "total_time_seconds": "time(s)",
    "error": "error",
}

# Comprehensive export fields
_COMPREHENSIVE_FIELDS = [
    "model_name", "precision", "mrr", "map",
    "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20",
    "recall@1", "recall@3", "recall@5", "recall@10", "recall@20",
    "precision@1", "precision@3", "precision@5", "precision@10", "precision@20",
    "f1@10", "r_precision", "hit_rate@10",
    "time(s)", "queries/sec", "error",
]


def export_json(results: list[ModelResult], output_path: Path) -> None:
    """Export results to JSON."""
    data = [asdict(r) for r in results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def export_comprehensive_json(
    results: list[ComprehensiveResult], output_path: Path
) -> None:
    """Export comprehensive results to JSON with all metrics."""
    data = []
    for r in results:
        entry: dict[str, Any] = {
            "model_name": r.model_name,
            "precision": r.precision,
            "status": r.status,
            "aggregate": {
                "mrr": r.mrr,
                "map": r.map,
                "r_precision": r.r_precision_stats,
            },
            "performance": {
                "total_time_seconds": r.total_time_seconds,
                "embedding_time_seconds": r.embedding_time_seconds,
                "num_corpus_chunks": r.num_corpus_chunks,
                "num_queries": r.num_queries,
                "queries_per_second": r.queries_per_second,
            },
        }

        # Add per-k metrics
        for k in [1, 3, 5, 10, 20]:
            if k in r.recall_stats:
                entry["aggregate"][f"recall@{k}"] = r.recall_stats[k]
            if k in r.precision_stats:
                entry["aggregate"][f"precision@{k}"] = r.precision_stats[k]
            if k in r.ndcg_stats:
                entry["aggregate"][f"ndcg@{k}"] = r.ndcg_stats[k]
            if k in r.f1_stats:
                entry["aggregate"][f"f1@{k}"] = r.f1_stats[k]
            if k in r.success_rates:
                entry["aggregate"][f"hit_rate@{k}"] = r.success_rates[k]

        if r.error:
            entry["error"] = r.error

        data.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def export_csv(results: list[ModelResult], output_path: Path) -> None:
    """Export results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in _FIELDS}
            writer.writerow(row)


def export_comprehensive_csv(
    results: list[ComprehensiveResult], output_path: Path
) -> None:
    """Export comprehensive results to CSV with all metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COMPREHENSIVE_FIELDS)
        writer.writeheader()
        for r in results:
            row = {
                "model_name": r.model_name,
                "precision": r.precision,
                "mrr": f"{r.mrr:.4f}",
                "map": f"{r.map:.4f}",
                "r_precision": f"{r.r_precision_stats.get('mean', 0.0):.4f}",
                "hit_rate@10": f"{r.success_rates.get(10, 0.0):.4f}",
                "time(s)": f"{r.total_time_seconds:.2f}",
                "queries/sec": f"{r.queries_per_second:.2f}",
                "error": r.error or "",
            }
            # Add per-k metrics (mean values)
            for k in [1, 3, 5, 10, 20]:
                row[f"ndcg@{k}"] = f"{r.ndcg_stats.get(k, {}).get('mean', 0.0):.4f}"
                row[f"recall@{k}"] = f"{r.recall_stats.get(k, {}).get('mean', 0.0):.4f}"
                row[f"precision@{k}"] = (
                    f"{r.precision_stats.get(k, {}).get('mean', 0.0):.4f}"
                )
            row["f1@10"] = f"{r.f1_stats.get(10, {}).get('mean', 0.0):.4f}"
            writer.writerow(row)


def export_markdown(results: list[ModelResult], output_path: Path) -> None:
    """Export results as a Markdown table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [_HEADER_LABELS[f] for f in _FIELDS]
    separator = "|".join(["---"] * len(headers))

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + separator + " |",
    ]
    for r in results:
        values = []
        for f in _FIELDS:
            val = getattr(r, f)
            if f in ("recall_at_10", "mrr", "ndcg_at_10", "precision_at_10"):
                values.append(f"{val:.4f}")
            elif f == "total_time_seconds":
                values.append(f"{val:.2f}")
            elif f == "error":
                values.append(val or "")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_comprehensive_markdown(
    results: list[ComprehensiveResult], output_path: Path
) -> None:
    """Export comprehensive results as a detailed Markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Embedding Model Benchmark Results\n"]

    for r in results:
        lines.append(f"## {r.model_name} ({r.precision})\n")

        if r.error:
            lines.append(f"**Error:** {r.error}\n")
            continue

        lines.append("### Ranking Quality Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| MRR | {r.mrr:.4f} |")
        lines.append(f"| MAP | {r.map:.4f} |")
        lines.append(
            f"| R-Precision | {r.r_precision_stats.get('mean', 0.0):.4f} "
            f"(±{r.r_precision_stats.get('std', 0.0):.4f}) |"
        )
        lines.append("")

        # NDCG table
        lines.append("### NDCG@k\n")
        lines.append("| k | Mean | Std | Min | Max |")
        lines.append("|---|------|-----|-----|-----|")
        for k in [1, 3, 5, 10, 20]:
            stats = r.ndcg_stats.get(k, {})
            lines.append(
                f"| {k} | {stats.get('mean', 0.0):.4f} | "
                f"{stats.get('std', 0.0):.4f} | "
                f"{stats.get('min', 0.0):.4f} | {stats.get('max', 0.0):.4f} |"
            )
        lines.append("")

        # Recall table
        lines.append("### Recall@k\n")
        lines.append("| k | Mean | Std | Min | Max |")
        lines.append("|---|------|-----|-----|-----|")
        for k in [1, 3, 5, 10, 20]:
            stats = r.recall_stats.get(k, {})
            lines.append(
                f"| {k} | {stats.get('mean', 0.0):.4f} | "
                f"{stats.get('std', 0.0):.4f} | "
                f"{stats.get('min', 0.0):.4f} | {stats.get('max', 0.0):.4f} |"
            )
        lines.append("")

        # Precision table
        lines.append("### Precision@k\n")
        lines.append("| k | Mean | Std | Min | Max |")
        lines.append("|---|------|-----|-----|-----|")
        for k in [1, 3, 5, 10, 20]:
            stats = r.precision_stats.get(k, {})
            lines.append(
                f"| {k} | {stats.get('mean', 0.0):.4f} | "
                f"{stats.get('std', 0.0):.4f} | "
                f"{stats.get('min', 0.0):.4f} | {stats.get('max', 0.0):.4f} |"
            )
        lines.append("")

        # Hit Rate table
        lines.append("### Hit Rate (Success@k)\n")
        lines.append("| k | Hit Rate |")
        lines.append("|---|----------|")
        for k in [1, 3, 5, 10, 20]:
            lines.append(f"| {k} | {r.success_rates.get(k, 0.0):.4f} |")
        lines.append("")

        # Performance
        lines.append("### Performance\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Time | {r.total_time_seconds:.2f}s |")
        lines.append(f"| Embedding Time | {r.embedding_time_seconds:.2f}s |")
        lines.append(f"| Queries/Second | {r.queries_per_second:.2f} |")
        lines.append(f"| Corpus Chunks | {r.num_corpus_chunks:,} |")
        lines.append(f"| Queries Evaluated | {r.num_queries:,} |")
        lines.append("\n---\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
