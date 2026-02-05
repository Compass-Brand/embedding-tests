"""Experiment checkpointing for resumable runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_checkpoint_path(
    checkpoint_dir: Path,
    model_name: str,
    precision: str,
) -> Path:
    """Get the checkpoint file path for a model/precision combination."""
    safe_name = model_name.replace("/", "_").replace(" ", "_")
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
    path.write_text(json.dumps(data, indent=2))
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
    return json.loads(path.read_text())


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
