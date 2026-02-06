"""Tests for cache module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestGetCacheDir:
    """Tests for get_cache_dir function."""

    def test_returns_default_cache_dir(self) -> None:
        """Test default cache directory is data/hf_cache."""
        from embedding_tests.config.cache import get_cache_dir

        with patch.dict(os.environ, {}, clear=True):
            # Clear EMB_TEST_DATA_DIR to use default
            os.environ.pop("EMB_TEST_DATA_DIR", None)
            cache_dir = get_cache_dir()
            assert cache_dir == Path("data") / "hf_cache"

    def test_respects_env_variable(self) -> None:
        """Test cache directory respects EMB_TEST_DATA_DIR env var."""
        from embedding_tests.config.cache import get_cache_dir

        with patch.dict(os.environ, {"EMB_TEST_DATA_DIR": "/custom/data"}):
            cache_dir = get_cache_dir()
            assert cache_dir == Path("/custom/data") / "hf_cache"

    def test_returns_path_object(self) -> None:
        """Test that get_cache_dir returns a Path object."""
        from embedding_tests.config.cache import get_cache_dir

        cache_dir = get_cache_dir()
        assert isinstance(cache_dir, Path)


class TestEnsureCacheDir:
    """Tests for ensure_cache_dir function."""

    def test_creates_directory_if_not_exists(self, tmp_path: Path) -> None:
        """Test that ensure_cache_dir creates the directory."""
        from embedding_tests.config.cache import ensure_cache_dir

        cache_dir = tmp_path / "cache" / "hf_cache"
        assert not cache_dir.exists()

        result = ensure_cache_dir(cache_dir)

        assert cache_dir.exists()
        assert cache_dir.is_dir()
        assert result == cache_dir

    def test_returns_existing_directory(self, tmp_path: Path) -> None:
        """Test that ensure_cache_dir returns existing directory."""
        from embedding_tests.config.cache import ensure_cache_dir

        cache_dir = tmp_path / "existing_cache"
        cache_dir.mkdir(parents=True)
        marker = cache_dir / "marker.txt"
        marker.write_text("test")

        result = ensure_cache_dir(cache_dir)

        assert result == cache_dir
        assert marker.exists()
