"""Dataset cache management.

Provides utilities for managing HuggingFace dataset cache location,
allowing datasets to be stored locally in data/hf_cache/ directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_cache_dir() -> Path:
    """Get HuggingFace cache directory from EMB_TEST_DATA_DIR or default.

    Returns:
        Path to the cache directory (data/hf_cache by default).
    """
    data_dir = Path(os.environ.get("EMB_TEST_DATA_DIR", "data"))
    return data_dir / "hf_cache"


def ensure_cache_dir(cache_dir: Path) -> Path:
    """Ensure the cache directory exists, creating it if necessary.

    Args:
        cache_dir: Path to the cache directory.

    Returns:
        The cache directory path (same as input).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
