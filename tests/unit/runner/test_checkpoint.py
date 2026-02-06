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
        assert loaded["model_name"] == "test-model"
        assert loaded["precision"] == "fp16"
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

    def test_load_checkpoint_returns_none_for_corrupted_file(self, tmp_path: Path) -> None:
        cp_path = tmp_path / "checkpoints"
        cp_path.mkdir(parents=True, exist_ok=True)
        path = get_checkpoint_path(cp_path, "test-model", "fp16")
        path.write_text("not valid json {{{")
        loaded = load_checkpoint(cp_path, model_name="test-model", precision="fp16")
        assert loaded is None

    def test_is_completed_returns_false_for_in_progress(self, tmp_path: Path) -> None:
        cp_path = tmp_path / "checkpoints"
        save_checkpoint(checkpoint_dir=cp_path, model_name="m1", precision="fp16", status="in_progress", results={})
        assert is_completed(cp_path, "m1", "fp16") is False

    def test_checkpoint_path_sanitizes_special_characters(self, tmp_path: Path) -> None:
        path = get_checkpoint_path(tmp_path, "org/model-name:v1", "fp16")
        assert "/" not in path.name
        assert ":" not in path.name
        assert path.suffix == ".json"

    def test_save_checkpoint_raises_on_write_failure(self, tmp_path: Path) -> None:
        """Test that OSError is raised when checkpoint write fails."""
        from unittest.mock import patch

        cp_path = tmp_path / "checkpoints"
        cp_path.mkdir(parents=True)

        with patch(
            "embedding_tests.runner.checkpoint.tempfile.mkstemp",
            side_effect=OSError("Disk full"),
        ):
            with pytest.raises(OSError, match="Failed to save checkpoint"):
                save_checkpoint(
                    checkpoint_dir=cp_path,
                    model_name="test-model",
                    precision="fp16",
                    status="completed",
                    results={},
                )

    def test_save_and_load_checkpoint_with_timing_and_mrr(self, tmp_path: Path) -> None:
        """Test that mrr and total_time are persisted and loaded correctly."""
        cp_path = tmp_path / "checkpoints"
        save_checkpoint(
            checkpoint_dir=cp_path,
            model_name="test-model",
            precision="fp16",
            status="completed",
            results={"recall_at_10": 0.9},
            mrr=0.85,
            total_time=1.234,
        )
        loaded = load_checkpoint(cp_path, model_name="test-model", precision="fp16")
        assert loaded is not None
        assert loaded["mrr"] == 0.85
        assert loaded["total_time"] == 1.234


class TestClearCheckpoints:
    """Tests for clear_checkpoints function."""

    def test_clear_checkpoints_removes_all_json_files(self, tmp_path: Path) -> None:
        """Test that clear_checkpoints removes all checkpoint JSON files."""
        from embedding_tests.runner.checkpoint import clear_checkpoints

        cp_path = tmp_path / "checkpoints"
        cp_path.mkdir(parents=True)

        # Create some checkpoint files
        (cp_path / "model1_fp16.json").write_text('{"status": "completed"}')
        (cp_path / "model2_int8.json").write_text('{"status": "completed"}')
        (cp_path / "model3_fp16.json").write_text('{"status": "failed"}')

        count = clear_checkpoints(cp_path)

        assert count == 3
        assert list(cp_path.glob("*.json")) == []

    def test_clear_checkpoints_returns_zero_for_empty_dir(self, tmp_path: Path) -> None:
        """Test that clear_checkpoints returns 0 for empty directory."""
        from embedding_tests.runner.checkpoint import clear_checkpoints

        cp_path = tmp_path / "checkpoints"
        cp_path.mkdir(parents=True)

        count = clear_checkpoints(cp_path)

        assert count == 0

    def test_clear_checkpoints_handles_nonexistent_dir(self, tmp_path: Path) -> None:
        """Test that clear_checkpoints handles non-existent directory."""
        from embedding_tests.runner.checkpoint import clear_checkpoints

        cp_path = tmp_path / "nonexistent"

        count = clear_checkpoints(cp_path)

        assert count == 0

    def test_clear_checkpoints_preserves_non_json_files(self, tmp_path: Path) -> None:
        """Test that clear_checkpoints only removes .json files."""
        from embedding_tests.runner.checkpoint import clear_checkpoints

        cp_path = tmp_path / "checkpoints"
        cp_path.mkdir(parents=True)

        # Create json and non-json files
        (cp_path / "model_fp16.json").write_text('{}')
        (cp_path / "notes.txt").write_text('keep me')
        (cp_path / "config.yaml").write_text('keep me too')

        count = clear_checkpoints(cp_path)

        assert count == 1
        assert (cp_path / "notes.txt").exists()
        assert (cp_path / "config.yaml").exists()
