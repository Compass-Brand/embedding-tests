"""Experiment checkpointing for resumable runs."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_checkpoint_path(
    checkpoint_dir: Path,
    model_name: str,
    precision: str,
) -> Path:
    """Get the checkpoint file path for a model/precision combination."""
    safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', model_name)
    return checkpoint_dir / f"{safe_name}_{precision}.json"


def save_checkpoint(
    checkpoint_dir: Path,
    model_name: str,
    precision: str,
    status: str,
    results: dict[str, Any],
    *,
    mrr: float | None = None,
    total_time: float | None = None,
) -> Path:
    """Save a checkpoint file."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = get_checkpoint_path(checkpoint_dir, model_name, precision)
    data: dict[str, Any] = {
        "model_name": model_name,
        "precision": precision,
        "status": status,
        "results": results,
    }
    if mrr is not None:
        data["mrr"] = mrr
    if total_time is not None:
        data["total_time"] = total_time
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=checkpoint_dir, suffix=".tmp", prefix=".checkpoint_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as e:
        raise OSError(f"Failed to save checkpoint to {path}: {e}") from e
    return path


def load_checkpoint(
    checkpoint_dir: Path,
    model_name: str,
    precision: str,
) -> dict[str, Any] | None:
    """Load a checkpoint if it exists."""
    path = get_checkpoint_path(checkpoint_dir, model_name, precision)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Corrupted checkpoint file: %s", path)
        return None
    except OSError as e:
        logger.warning("Failed to read checkpoint file %s: %s", path, e)
        return None


def is_completed(
    checkpoint_dir: Path,
    model_name: str,
    precision: str,
) -> bool:
    """Check if a run has been completed."""
    checkpoint = load_checkpoint(checkpoint_dir, model_name, precision)
    if checkpoint is None:
        return False
    return checkpoint.get("status") == "completed"


def clear_checkpoints(checkpoint_dir: Path) -> int:
    """Clear all checkpoint files from a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files.

    Returns:
        Number of checkpoint files deleted.
    """
    if not checkpoint_dir.exists():
        return 0

    count = 0
    for cp_file in checkpoint_dir.glob("*.json"):
        try:
            cp_file.unlink()
            count += 1
        except OSError as e:
            logger.warning("Failed to delete checkpoint %s: %s", cp_file, e)

    return count
