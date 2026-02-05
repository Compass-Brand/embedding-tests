"""Tests for model loader factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from embedding_tests.config.models import ModelConfig, ModelType, PrecisionLevel
from embedding_tests.hardware.precision import PrecisionConfig
from embedding_tests.models.loader import load_model


@pytest.fixture
def fp16_precision() -> PrecisionConfig:
    return PrecisionConfig(
        storage_dtype="float16",
        compute_dtype="float32",
        attn_implementation="eager",
        use_autocast=False,
    )


def _make_config(model_type: ModelType) -> ModelConfig:
    return ModelConfig(
        name="test",
        model_id="org/test",
        model_type=model_type,
        params_billions=0.6,
        embedding_dim=1024 if model_type != ModelType.MULTIMODAL_RERANKER else 0,
        supported_precisions=[PrecisionLevel.FP16],
    )


class TestModelLoader:
    """Tests for model loader factory."""

    @patch("embedding_tests.models.loader.SentenceTransformerWrapper")
    def test_loader_dispatches_text_embedding_to_st_wrapper(
        self, mock_st: MagicMock, fp16_precision: PrecisionConfig
    ) -> None:
        config = _make_config(ModelType.TEXT_EMBEDDING)
        load_model(config, fp16_precision)
        mock_st.assert_called_once_with(config, fp16_precision)

    @patch("embedding_tests.models.loader.VLEmbeddingWrapper")
    def test_loader_dispatches_multimodal_embedding_to_vl_wrapper(
        self, mock_vl: MagicMock, fp16_precision: PrecisionConfig
    ) -> None:
        config = _make_config(ModelType.MULTIMODAL_EMBEDDING)
        load_model(config, fp16_precision)
        mock_vl.assert_called_once_with(config, fp16_precision)

    @patch("embedding_tests.models.loader.VLRerankerWrapper")
    def test_loader_dispatches_multimodal_reranker_to_vl_reranker(
        self, mock_vl_reranker: MagicMock, fp16_precision: PrecisionConfig
    ) -> None:
        config = _make_config(ModelType.MULTIMODAL_RERANKER)
        load_model(config, fp16_precision)
        mock_vl_reranker.assert_called_once_with(config, fp16_precision)

    @patch("embedding_tests.models.loader.SentenceTransformerWrapper")
    def test_loader_passes_precision_config_through(
        self, mock_st: MagicMock, fp16_precision: PrecisionConfig
    ) -> None:
        config = _make_config(ModelType.TEXT_EMBEDDING)
        load_model(config, fp16_precision)
        # Precision config passed as positional arg
        assert mock_st.call_args.args[1] == fp16_precision
