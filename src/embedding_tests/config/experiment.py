"""Experiment configuration system."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from embedding_tests.config.models import (
    ModelConfig,
    PrecisionLevel,
    load_all_model_configs,
)

_VALID_PRECISIONS = {p.value for p in PrecisionLevel}


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the RAG pipeline within an experiment."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 10
    reranker_top_k: int = 3
    similarity_metric: str = "cosine"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single experiment run."""

    name: str
    description: str
    models: Sequence[ModelConfig]
    precisions: Sequence[PrecisionLevel]
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    reranker: str | None = None
    datasets: Sequence[str] = field(default_factory=tuple)
    mteb_tasks: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", tuple(self.models))
        object.__setattr__(self, "precisions", tuple(self.precisions))
        object.__setattr__(self, "datasets", tuple(self.datasets))
        object.__setattr__(self, "mteb_tasks", tuple(self.mteb_tasks))
        if not self.models:
            raise ValueError("models must not be empty")
        if not self.precisions:
            raise ValueError("precisions must not be empty")


def load_experiment_config(
    experiment_path: Path,
    models_dir: Path,
) -> ExperimentConfig:
    """Load an experiment config from YAML, resolving model references."""
    with open(experiment_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty or invalid YAML file: {experiment_path}")

    # Validate name field early
    if "name" not in data:
        raise ValueError("Experiment config must have a 'name' field")

    # Load all available model configs
    all_models = load_all_model_configs(models_dir)
    model_lookup: dict[str, ModelConfig] = {}
    for m in all_models:
        if m.name in model_lookup:
            raise ValueError(f"Duplicate model config name: {m.name!r}")
        model_lookup[m.name] = m

    # Resolve model references
    resolved_models: list[ModelConfig] = []
    for model_name in data.get("models", []):
        if model_name not in model_lookup:
            available = sorted(model_lookup.keys())
            raise ValueError(
                f"Unknown model: {model_name!r}. Available: {available}"
            )
        resolved_models.append(model_lookup[model_name])

    # Validate precisions
    precisions: list[PrecisionLevel] = []
    for p in data.get("precisions", []):
        if p not in _VALID_PRECISIONS:
            raise ValueError(
                f"Unknown precision: {p!r}. Valid: {sorted(_VALID_PRECISIONS)}"
            )
        precisions.append(PrecisionLevel(p))

    # Parse pipeline config
    pipeline_data = data.get("pipeline", {})
    if pipeline_data:
        known_fields = {f.name for f in fields(PipelineConfig)}
        unknown = set(pipeline_data.keys()) - known_fields
        if unknown:
            raise ValueError(f"Unknown pipeline fields: {sorted(unknown)}. Valid: {sorted(known_fields)}")
        pipeline = PipelineConfig(**pipeline_data)
    else:
        pipeline = PipelineConfig()

    return ExperimentConfig(
        name=data["name"],
        description=data.get("description", ""),
        models=resolved_models,
        precisions=precisions,
        pipeline=pipeline,
        reranker=data.get("reranker"),
        datasets=tuple(data.get("datasets", [])),
        mteb_tasks=tuple(data.get("mteb_tasks", [])),
    )
