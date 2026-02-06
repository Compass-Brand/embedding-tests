"""Shared integration test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from embedding_tests.config.hardware import GpuCapabilities, detect_gpu


# Navigate from tests/integration/conftest.py to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def gpu() -> GpuCapabilities:
    """Detect GPU and skip if not available."""
    detected_gpu = detect_gpu()
    if detected_gpu is None:
        pytest.skip("No CUDA GPU available")
    return detected_gpu


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    """Path to configs directory."""
    return _PROJECT_ROOT / "configs"


@pytest.fixture
def sample_corpus() -> list[dict[str, str]]:
    """Sample corpus for integration tests."""
    return [
        {"doc_id": f"doc_{i}", "text": f"Sample document {i} with content about topic {i % 3}."}
        for i in range(10)
    ]


@pytest.fixture
def sample_queries() -> list[dict]:
    """Sample queries with relevance judgments."""
    return [
        {"query_id": "q1", "text": "What is topic 0?", "relevant_doc_ids": ["doc_0", "doc_3", "doc_6", "doc_9"]},
        {"query_id": "q2", "text": "What is topic 1?", "relevant_doc_ids": ["doc_1", "doc_4", "doc_7"]},
        {"query_id": "q3", "text": "What is topic 2?", "relevant_doc_ids": ["doc_2", "doc_5", "doc_8"]},
    ]
