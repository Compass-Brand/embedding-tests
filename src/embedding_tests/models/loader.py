"""Model loader factory with hardware-aware dispatch."""

from __future__ import annotations

from embedding_tests.config.models import ModelConfig, ModelType
from embedding_tests.hardware.precision import PrecisionConfig
from embedding_tests.models.base import EmbeddingModel, RerankerModel
from embedding_tests.models.st_wrapper import SentenceTransformerWrapper
from embedding_tests.models.vl_embedding_wrapper import VLEmbeddingWrapper
from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper


def load_model(
    config: ModelConfig,
    precision: PrecisionConfig,
) -> EmbeddingModel | RerankerModel:
    """Load a model based on its type, dispatching to the correct wrapper."""
    if config.model_type == ModelType.TEXT_EMBEDDING:
        return SentenceTransformerWrapper(config, precision)
    if config.model_type == ModelType.MULTIMODAL_EMBEDDING:
        return VLEmbeddingWrapper(config, precision)
    if config.model_type == ModelType.MULTIMODAL_RERANKER:
        return VLRerankerWrapper(config, precision)
    raise ValueError(f"Unsupported model type: {config.model_type}")
