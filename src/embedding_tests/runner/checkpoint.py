"""Experiment checkpointing for resumable runs."""

from __future__ import annotations

import json
import logging
import re
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
) -> Path:
    """Save a checkpoint file."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = get_checkpoint_path(checkpoint_dir, model_name, precision)
    data = {
        "model_name": model_name,
        "precision": precision,
        "status": status,
        "results": results,
    }
    try:
        path.write_text(json.dumps(data, indent=2))
    except OSError as e:
        raise IOError(f"Failed to save checkpoint to {path}: {e}") from e
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
