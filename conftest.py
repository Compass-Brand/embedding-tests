"""Root conftest for embedding-tests. Registers pytest markers and GPU skip logic."""

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: requires GPU hardware")
    config.addinivalue_line("markers", "slow: slow running tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "vl: requires VL model dependencies")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip GPU tests when no CUDA device is available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
