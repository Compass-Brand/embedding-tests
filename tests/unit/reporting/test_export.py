"""Tests for multi-format export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embedding_tests.reporting.collector import ModelResult
from embedding_tests.reporting.export import export_json, export_csv, export_markdown


@pytest.fixture
def sample_results() -> list[ModelResult]:
    return [
        ModelResult("m1", "fp16", 0.8, 0.7, 0.75, 0.6, 10.0),
        ModelResult("m2", "fp16", 0.9, 0.8, 0.85, 0.7, 15.0),
        ModelResult("m3", "fp16", 0.0, 0.0, 0.0, 0.0, 0.0, error="Model load failed"),
    ]


class TestExportJSON:
    """Tests for JSON export."""

    def test_export_json_valid_structure(
        self, sample_results: list[ModelResult], tmp_path: Path
    ) -> None:
        output = tmp_path / "results.json"
        export_json(sample_results, output)
        data = json.loads(output.read_text())
        assert isinstance(data, list)
        assert len(data) == 3
        assert "model_name" in data[0]
        assert data[0]["model_name"] == "m1"
        assert data[0]["recall_at_10"] == 0.8
        assert data[1]["model_name"] == "m2"
        assert data[2]["model_name"] == "m3"
        assert data[2]["error"] == "Model load failed"


class TestExportCSV:
    """Tests for CSV export."""

    def test_export_csv_correct_headers_and_rows(
        self, sample_results: list[ModelResult], tmp_path: Path
    ) -> None:
        output = tmp_path / "results.csv"
        export_csv(sample_results, output)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        header = lines[0]
        assert "model_name" in header
        assert "recall_at_10" in header

    def test_export_csv_data_content_matches_results(
        self, sample_results: list[ModelResult], tmp_path: Path
    ) -> None:
        output = tmp_path / "results.csv"
        export_csv(sample_results, output)
        with open(output, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["model_name"] == "m1"
        assert rows[0]["precision"] == "fp16"
        assert float(rows[0]["recall_at_10"]) == 0.8
        assert float(rows[0]["mrr"]) == 0.7
        assert float(rows[0]["ndcg_at_10"]) == 0.75
        assert float(rows[0]["precision_at_10"]) == 0.6
        assert float(rows[0]["total_time_seconds"]) == 10.0
        assert rows[1]["model_name"] == "m2"
        assert float(rows[1]["recall_at_10"]) == 0.9
        assert rows[2]["model_name"] == "m3"
        assert rows[2]["error"] == "Model load failed"


class TestExportMarkdown:
    """Tests for Markdown export."""

    def test_export_markdown_table_formatting(
        self, sample_results: list[ModelResult], tmp_path: Path
    ) -> None:
        output = tmp_path / "results.md"
        export_markdown(sample_results, output)
        content = output.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 5  # header + separator + 3 data rows
        assert "---" in lines[1]
        assert "model_name" in lines[0]
