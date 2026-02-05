"""Tests for experiment checkpoint system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embedding_tests.runner.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    is_completed,
    get_checkpoint_path,
)


class TestCheckpoint:
    """Tests for checkpoint system."""

    def test_save_checkpoint_creates_json_file(self, tmp_path: Path) -> None:
        cp_path = tmp_path / "checkpoints"
        save_checkpoint(
            checkpoint_dir=cp_path,
            model_name="test-model",
            precision="fp16",
            status="completed",
            results={"recall_at_10": 0.85},
        )
        files = list(cp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["status"] == "completed"

    def test_load_checkpoint_restores_state(self, tmp_path: Path) -> None:
        cp_path = tmp_path / "checkpoints"
        save_checkpoint(
            checkpoint_dir=cp_path,
            model_name="test-model",
            precision="fp16",
            status="completed",
            results={"score": 0.9},
        )
        loaded = load_checkpoint(cp_path, model_name="test-model", precision="fp16")
        assert loaded is not None
        assert loaded["status"] == "completed"
        assert loaded["results"]["score"] == 0.9

    def test_is_completed_returns_true_for_finished_runs(self, tmp_path: Path) -> None:
        cp_path = tmp_path / "checkpoints"
        save_checkpoint(
            checkpoint_dir=cp_path,
            model_name="m1",
            precision="fp16",
            status="completed",
            results={},
        )
        assert is_completed(cp_path, "m1", "fp16") is True

    def test_is_completed_returns_false_for_missing(self, tmp_path: Path) -> None:
        cp_path = tmp_path / "checkpoints"
        cp_path.mkdir(parents=True, exist_ok=True)
        assert is_completed(cp_path, "nonexistent", "fp16") is False

    def test_checkpoint_path_includes_model_precision(self, tmp_path: Path) -> None:
        path = get_checkpoint_path(tmp_path, "my-model", "int8")
        assert "my-model" in path.name
        assert "int8" in path.name
        assert path.suffix == ".json"
