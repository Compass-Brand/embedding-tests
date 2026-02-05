"""Multi-format result export (JSON, CSV, Markdown)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from embedding_tests.reporting.collector import ModelResult

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


def export_json(results: list[ModelResult], output_path: Path) -> None:
    """Export results to JSON."""
    data = [asdict(r) for r in results]
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
