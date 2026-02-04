"""Shared test fixtures for embedding-tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_corpus(fixtures_dir: Path) -> list[dict[str, Any]]:
    """Load sample corpus from fixtures."""
    import json

    corpus_path = fixtures_dir / "sample_corpus.json"
    with open(corpus_path) as f:
        return json.load(f)


@pytest.fixture
def sample_queries(fixtures_dir: Path) -> list[dict[str, Any]]:
    """Load sample queries with ground truth from fixtures."""
    import json

    queries_path = fixtures_dir / "sample_queries.json"
    with open(queries_path) as f:
        return json.load(f)


@pytest.fixture
def configs_dir() -> Path:
    """Return path to the configs directory."""
    return Path(__file__).parent.parent / "configs"


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test results."""
    results = tmp_path / "results"
    results.mkdir()
    return results
