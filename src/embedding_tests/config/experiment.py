"""Experiment configuration system."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    models: list[ModelConfig]
    precisions: list[PrecisionLevel]
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    reranker: str | None = None
    datasets: list[str] = field(default_factory=list)


def load_experiment_config(
    experiment_path: Path,
    models_dir: Path,
) -> ExperimentConfig:
    """Load an experiment config from YAML, resolving model references."""
    with open(experiment_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Validate name field early
    if "name" not in data:
        raise ValueError("Experiment config must have a 'name' field")

    # Load all available model configs
    all_models = load_all_model_configs(models_dir)
    model_lookup = {m.name: m for m in all_models}

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
    pipeline = PipelineConfig(**pipeline_data) if pipeline_data else PipelineConfig()

    return ExperimentConfig(
        name=data["name"],
        description=data.get("description", ""),
        models=resolved_models,
        precisions=precisions,
        pipeline=pipeline,
        reranker=data.get("reranker"),
        datasets=data.get("datasets", []),
    )
