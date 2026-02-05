"""Multi-format result export (JSON, CSV, Markdown)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from embedding_tests.reporting.collector import ModelResult

_FIELDS = [
    "model_name", "precision", "recall_at_10", "mrr",
    "ndcg_at_10", "precision_at_10", "total_time_seconds",
]


def export_json(results: list[ModelResult], output_path: Path) -> None:
    """Export results to JSON."""
    data = [asdict(r) for r in results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


def export_csv(results: list[ModelResult], output_path: Path) -> None:
    """Export results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in _FIELDS}
            writer.writerow(row)


def export_markdown(results: list[ModelResult], output_path: Path) -> None:
    """Export results as a Markdown table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["model_name", "precision", "recall@10", "MRR", "NDCG@10", "P@10", "time(s)"]
    separator = "|".join(["---"] * len(headers))

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + separator + " |",
    ]
    for r in results:
        row = [
            r.model_name, r.precision,
            f"{r.recall_at_10:.4f}", f"{r.mrr:.4f}",
            f"{r.ndcg_at_10:.4f}", f"{r.precision_at_10:.4f}",
            f"{r.total_time_seconds:.2f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    output_path.write_text("\n".join(lines) + "\n")
