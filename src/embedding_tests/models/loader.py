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
    logger.debug(
        "Loading model %s (%s) with precision %s",
        config.name,
        config.model_type.value,
        precision.storage_dtype,
    )
    match config.model_type:
        case ModelType.TEXT_EMBEDDING:
            return SentenceTransformerWrapper(config, precision)
        case ModelType.MULTIMODAL_EMBEDDING:
            return VLEmbeddingWrapper(config, precision)
        case ModelType.MULTIMODAL_RERANKER:
            return VLRerankerWrapper(config, precision)
        case _:
            raise ValueError(f"Unsupported model type: {config.model_type}")
