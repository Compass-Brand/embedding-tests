"""Tests for sentence-transformers wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from embedding_tests.config.models import ModelConfig, ModelType, PrecisionLevel
from embedding_tests.hardware.precision import PrecisionConfig
from embedding_tests.models.base import EmbeddingModel


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        name="test-model",
        model_id="org/test-model",
        model_type=ModelType.TEXT_EMBEDDING,
        params_billions=0.6,
        embedding_dim=1024,
        supported_precisions=[PrecisionLevel.FP16],
        trust_remote_code=True,
        query_instruction="query: ",
        padding_side="left",
    )


@pytest.fixture
def fp16_precision() -> PrecisionConfig:
    return PrecisionConfig(
        storage_dtype="float16",
        compute_dtype="float32",
        attn_implementation="eager",
        use_autocast=False,
    )


@pytest.fixture
def int8_precision() -> PrecisionConfig:
    return PrecisionConfig(
        storage_dtype="int8",
        compute_dtype="float32",
        attn_implementation="eager",
        use_autocast=False,
        quantization_config={"load_in_8bit": True, "llm_int8_threshold": 6.0},
    )


class TestSTWrapper:
    """Tests for SentenceTransformerWrapper."""

    @patch("embedding_tests.models.st_wrapper.SentenceTransformer")
    def test_st_wrapper_init_sets_fp16_dtype(
        self, mock_st_cls: MagicMock, model_config: ModelConfig, fp16_precision: PrecisionConfig
    ) -> None:
        from embedding_tests.models.st_wrapper import SentenceTransformerWrapper

        SentenceTransformerWrapper(model_config, fp16_precision)
        call_kwargs = mock_st_cls.call_args
        model_kwargs = call_kwargs.kwargs.get("model_kwargs", {})
        assert model_kwargs.get("torch_dtype") == torch.float16

    @patch("embedding_tests.models.st_wrapper.SentenceTransformer")
    def test_st_wrapper_init_sets_eager_attention(
        self, mock_st_cls: MagicMock, model_config: ModelConfig, fp16_precision: PrecisionConfig
    ) -> None:
        from embedding_tests.models.st_wrapper import SentenceTransformerWrapper

        SentenceTransformerWrapper(model_config, fp16_precision)
        call_kwargs = mock_st_cls.call_args
        model_kwargs = call_kwargs.kwargs.get("model_kwargs", {})
        assert model_kwargs.get("attn_implementation") == "eager"

    @patch("embedding_tests.models.st_wrapper.SentenceTransformer")
    def test_st_wrapper_init_sets_trust_remote_code(
        self, mock_st_cls: MagicMock, model_config: ModelConfig, fp16_precision: PrecisionConfig
    ) -> None:
        from embedding_tests.models.st_wrapper import SentenceTransformerWrapper

        SentenceTransformerWrapper(model_config, fp16_precision)
        call_kwargs = mock_st_cls.call_args
        assert call_kwargs.kwargs.get("trust_remote_code") is True

    @patch("embedding_tests.models.st_wrapper.SentenceTransformer")
    def test_st_wrapper_encode_returns_normalized_embeddings(
        self, mock_st_cls: MagicMock, model_config: ModelConfig, fp16_precision: PrecisionConfig
    ) -> None:
        from embedding_tests.models.st_wrapper import SentenceTransformerWrapper

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_st_cls.return_value = mock_model

        wrapper = SentenceTransformerWrapper(model_config, fp16_precision)
        wrapper.encode(["test text"])

        encode_call = mock_model.encode.call_args
        assert encode_call.kwargs.get("normalize_embeddings") is True

    @patch("embedding_tests.models.st_wrapper.SentenceTransformer")
    def test_st_wrapper_encode_query_uses_instruction(
        self, mock_st_cls: MagicMock, model_config: ModelConfig, fp16_precision: PrecisionConfig
    ) -> None:
        from embedding_tests.models.st_wrapper import SentenceTransformerWrapper

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0]])
        mock_st_cls.return_value = mock_model

        wrapper = SentenceTransformerWrapper(model_config, fp16_precision)
        wrapper.encode(["test"], is_query=True)

        encode_call = mock_model.encode.call_args
        assert encode_call.kwargs.get("prompt") == "query: "

    @patch("embedding_tests.models.st_wrapper.SentenceTransformer")
    def test_st_wrapper_encode_document_no_instruction(
        self, mock_st_cls: MagicMock, model_config: ModelConfig, fp16_precision: PrecisionConfig
    ) -> None:
        from embedding_tests.models.st_wrapper import SentenceTransformerWrapper

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0]])
        mock_st_cls.return_value = mock_model

        wrapper = SentenceTransformerWrapper(model_config, fp16_precision)
        wrapper.encode(["test"], is_query=False)

        encode_call = mock_model.encode.call_args
        assert encode_call.kwargs.get("prompt") is None

    @patch("embedding_tests.models.st_wrapper.SentenceTransformer")
    def test_st_wrapper_int8_loading_uses_bitsandbytes(
        self, mock_st_cls: MagicMock, model_config: ModelConfig, int8_precision: PrecisionConfig
    ) -> None:
        from embedding_tests.models.st_wrapper import SentenceTransformerWrapper

        SentenceTransformerWrapper(model_config, int8_precision)
        call_kwargs = mock_st_cls.call_args
        model_kwargs = call_kwargs.kwargs.get("model_kwargs", {})
        quant = model_kwargs.get("quantization_config")
        assert quant is not None

    @patch("embedding_tests.models.st_wrapper.torch")
    @patch("embedding_tests.models.st_wrapper.SentenceTransformer")
    def test_st_wrapper_unload_clears_memory(
        self,
        mock_st_cls: MagicMock,
        mock_torch: MagicMock,
        model_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.st_wrapper import SentenceTransformerWrapper

        mock_torch.cuda.is_available.return_value = True
        wrapper = SentenceTransformerWrapper(model_config, fp16_precision)
        wrapper.unload()
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("embedding_tests.models.st_wrapper.SentenceTransformer")
    def test_st_wrapper_satisfies_embedding_protocol(
        self, mock_st_cls: MagicMock, model_config: ModelConfig, fp16_precision: PrecisionConfig
    ) -> None:
        from embedding_tests.models.st_wrapper import SentenceTransformerWrapper

        wrapper = SentenceTransformerWrapper(model_config, fp16_precision)
        assert isinstance(wrapper, EmbeddingModel)

    @patch("embedding_tests.models.st_wrapper.SentenceTransformer")
    def test_st_wrapper_get_embedding_dim(
        self, mock_st_cls: MagicMock, model_config: ModelConfig, fp16_precision: PrecisionConfig
    ) -> None:
        from embedding_tests.models.st_wrapper import SentenceTransformerWrapper

        wrapper = SentenceTransformerWrapper(model_config, fp16_precision)
        assert wrapper.get_embedding_dim() == 1024
