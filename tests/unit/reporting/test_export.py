"""Tests for multi-format export."""

from __future__ import annotations

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
        assert len(data) == 2
        assert "model_name" in data[0]
        assert data[0]["model_name"] == "m1"
        assert data[0]["recall_at_10"] == 0.8
        assert data[1]["model_name"] == "m2"


class TestExportCSV:
    """Tests for CSV export."""

    def test_export_csv_correct_headers_and_rows(
        self, sample_results: list[ModelResult], tmp_path: Path
    ) -> None:
        output = tmp_path / "results.csv"
        export_csv(sample_results, output)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        header = lines[0]
        assert "model_name" in header
        assert "recall_at_10" in header


class TestExportMarkdown:
    """Tests for Markdown export."""

    def test_export_markdown_table_formatting(
        self, sample_results: list[ModelResult], tmp_path: Path
    ) -> None:
        output = tmp_path / "results.md"
        export_markdown(sample_results, output)
        content = output.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 4  # header + separator + 2 data rows
        assert "---" in lines[1]
        assert "model_name" in lines[0]
