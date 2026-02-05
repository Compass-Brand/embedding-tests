"""Shared integration test fixtures."""

from __future__ import annotations

import pytest

from embedding_tests.config.hardware import GpuCapabilities, detect_gpu


@pytest.fixture
def gpu() -> GpuCapabilities:
    """Detect GPU and skip if not available."""
    detected_gpu = detect_gpu()
    if detected_gpu is None:
        pytest.skip("No CUDA GPU available")
    return detected_gpu
