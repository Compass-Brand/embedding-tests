"""Root conftest for embedding-tests. Registers pytest markers and GPU skip logic."""

import functools

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: requires GPU hardware")
    config.addinivalue_line("markers", "slow: slow running tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "vl: requires VL model dependencies")


@functools.lru_cache(maxsize=1)
def _gpu_is_usable() -> bool:
    """Check if CUDA GPU is available and compatible with this PyTorch build."""
    if not torch.cuda.is_available():
        return False
    try:
        # Verify the GPU is actually usable by attempting a small tensor op
        t = torch.zeros(1, device="cuda")
        del t
        torch.cuda.empty_cache()
        return True
    except Exception:  # broad catch: driver mismatch, OOM, etc.
        return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip GPU tests when no usable CUDA device is available."""
    if not _gpu_is_usable():
        skip_gpu = pytest.mark.skip(reason="No usable CUDA GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
