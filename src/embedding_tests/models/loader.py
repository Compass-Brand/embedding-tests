"""Model loader factory with hardware-aware dispatch."""

from __future__ import annotations

import logging

from embedding_tests.config.models import ModelConfig, ModelType
from embedding_tests.hardware.precision import PrecisionConfig
from embedding_tests.models.base import EmbeddingModel, RerankerModel
from embedding_tests.models.st_wrapper import SentenceTransformerWrapper
from embedding_tests.models.vl_embedding_wrapper import VLEmbeddingWrapper
from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

logger = logging.getLogger(__name__)


def load_model(
    config: ModelConfig,
    precision: PrecisionConfig,
) -> EmbeddingModel | RerankerModel:
    """Load a model based on its type, dispatching to the correct wrapper."""
    model_type_str = config.model_type.value if isinstance(config.model_type, ModelType) else str(config.model_type)
    logger.debug(
        "Loading model %s (%s) with precision %s",
        config.name,
        model_type_str,
        precision.storage_dtype,
    )
    if config.model_type == ModelType.TEXT_EMBEDDING:
        return SentenceTransformerWrapper(config, precision)
    if config.model_type == ModelType.MULTIMODAL_EMBEDDING:
        return VLEmbeddingWrapper(config, precision)
    if config.model_type == ModelType.MULTIMODAL_RERANKER:
        return VLRerankerWrapper(config, precision)
    raise ValueError(f"Unsupported model type: {config.model_type}")
