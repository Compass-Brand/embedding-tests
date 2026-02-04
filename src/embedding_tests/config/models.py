"""Model configuration system for embedding and reranker models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml


class ModelType(Enum):
    """Types of models supported by the framework."""

    TEXT_EMBEDDING = "text_embedding"
    MULTIMODAL_EMBEDDING = "multimodal_embedding"
    MULTIMODAL_RERANKER = "multimodal_reranker"


class PrecisionLevel(Enum):
    """Supported precision levels for model loading."""

    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ_INT4 = "gptq_int4"
    AWQ_INT4 = "awq_int4"


_VALID_PRECISIONS = {p.value for p in PrecisionLevel}


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model."""

    name: str
    model_id: str
    model_type: ModelType
    params_billions: float
    embedding_dim: int
    supported_precisions: list[PrecisionLevel]
    trust_remote_code: bool = True
    query_instruction: str | None = None
    document_instruction: str | None = None
    max_seq_length: int | None = None
    padding_side: str | None = None
    extra_kwargs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Model name cannot be empty")
        if self.params_billions <= 0:
            raise ValueError(f"params_billions must be positive, got {self.params_billions}")
        if self.model_type != ModelType.MULTIMODAL_RERANKER and self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.embedding_dim < 0:
            raise ValueError(f"embedding_dim cannot be negative, got {self.embedding_dim}")


def load_model_config(path: Path) -> ModelConfig:
    """Load a ModelConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    model_type = ModelType(data["model_type"])

    precisions: list[PrecisionLevel] = []
    for p in data["supported_precisions"]:
        if p not in _VALID_PRECISIONS:
            raise ValueError(f"Unknown precision: {p!r}. Valid: {sorted(_VALID_PRECISIONS)}")
        precisions.append(PrecisionLevel(p))

    return ModelConfig(
        name=data["name"],
        model_id=data["model_id"],
        model_type=model_type,
        params_billions=data["params_billions"],
        embedding_dim=data["embedding_dim"],
        supported_precisions=precisions,
        trust_remote_code=data.get("trust_remote_code", True),
        query_instruction=data.get("query_instruction"),
        document_instruction=data.get("document_instruction"),
        max_seq_length=data.get("max_seq_length"),
        padding_side=data.get("padding_side"),
        extra_kwargs=data.get("extra_kwargs", {}),
    )


def load_all_model_configs(models_dir: Path) -> list[ModelConfig]:
    """Load all model configs from a directory of YAML files."""
    configs: list[ModelConfig] = []
    for yaml_file in sorted(models_dir.glob("*.yaml")):
        configs.append(load_model_config(yaml_file))
    return configs
