"""Tests for VL embedding wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from embedding_tests.config.models import ModelConfig, ModelType, PrecisionLevel
from embedding_tests.hardware.precision import PrecisionConfig
from embedding_tests.models.base import EmbeddingModel


@pytest.fixture
def vl_model_config() -> ModelConfig:
    return ModelConfig(
        name="qwen3-vl-embedding-2b",
        model_id="Qwen/Qwen3-VL-Embedding-2B",
        model_type=ModelType.MULTIMODAL_EMBEDDING,
        params_billions=2.0,
        embedding_dim=2048,
        supported_precisions=[PrecisionLevel.FP16],
        trust_remote_code=True,
    )


@pytest.fixture
def fp16_precision() -> PrecisionConfig:
    return PrecisionConfig(
        storage_dtype="float16",
        compute_dtype="float32",
        attn_implementation="eager",
        use_autocast=False,
    )


class TestVLEmbeddingWrapper:
    """Tests for VLEmbeddingWrapper."""

    @patch("embedding_tests.models.vl_embedding_wrapper.AutoTokenizer")
    @patch("embedding_tests.models.vl_embedding_wrapper.AutoModel")
    def test_vl_emb_init_uses_fp16_not_bf16(
        self,
        mock_auto_model: MagicMock,
        mock_tokenizer: MagicMock,
        vl_model_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_embedding_wrapper import VLEmbeddingWrapper

        VLEmbeddingWrapper(vl_model_config, fp16_precision)
        call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
        assert call_kwargs.get("torch_dtype") == torch.float16

    @patch("embedding_tests.models.vl_embedding_wrapper.AutoTokenizer")
    @patch("embedding_tests.models.vl_embedding_wrapper.AutoModel")
    def test_vl_emb_init_uses_eager_attention(
        self,
        mock_auto_model: MagicMock,
        mock_tokenizer: MagicMock,
        vl_model_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_embedding_wrapper import VLEmbeddingWrapper

        VLEmbeddingWrapper(vl_model_config, fp16_precision)
        call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
        assert call_kwargs.get("attn_implementation") == "eager"

    @patch("embedding_tests.models.vl_embedding_wrapper.AutoTokenizer")
    @patch("embedding_tests.models.vl_embedding_wrapper.AutoModel")
    def test_vl_emb_encode_text_returns_correct_shape(
        self,
        mock_auto_model: MagicMock,
        mock_tokenizer: MagicMock,
        vl_model_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_embedding_wrapper import VLEmbeddingWrapper

        mock_model = MagicMock()
        # Return real tensors so the encode pooling math works
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
        )
        mock_model.return_value = mock_output
        mock_model.get_input_embeddings.return_value.weight.device = torch.device("cpu")
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }
        mock_tokenizer.from_pretrained.return_value = mock_tok

        wrapper = VLEmbeddingWrapper(vl_model_config, fp16_precision)
        result = wrapper.encode(["hello", "world"])
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2  # Two input texts

    @patch("embedding_tests.models.vl_embedding_wrapper.torch")
    @patch("embedding_tests.models.vl_embedding_wrapper.AutoTokenizer")
    @patch("embedding_tests.models.vl_embedding_wrapper.AutoModel")
    def test_vl_emb_unload_clears_memory(
        self,
        mock_auto_model: MagicMock,
        mock_tokenizer: MagicMock,
        mock_torch: MagicMock,
        vl_model_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_embedding_wrapper import VLEmbeddingWrapper

        mock_torch.cuda.is_available.return_value = True
        wrapper = VLEmbeddingWrapper(vl_model_config, fp16_precision)
        wrapper.unload()
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("embedding_tests.models.vl_embedding_wrapper.AutoTokenizer")
    @patch("embedding_tests.models.vl_embedding_wrapper.AutoModel")
    def test_vl_emb_satisfies_embedding_protocol(
        self,
        mock_auto_model: MagicMock,
        mock_tokenizer: MagicMock,
        vl_model_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_embedding_wrapper import VLEmbeddingWrapper

        wrapper = VLEmbeddingWrapper(vl_model_config, fp16_precision)
        assert isinstance(wrapper, EmbeddingModel)

    @patch("embedding_tests.models.vl_embedding_wrapper.AutoTokenizer")
    @patch("embedding_tests.models.vl_embedding_wrapper.AutoModel")
    def test_vl_emb_get_embedding_dim(
        self,
        mock_auto_model: MagicMock,
        mock_tokenizer: MagicMock,
        vl_model_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_embedding_wrapper import VLEmbeddingWrapper

        wrapper = VLEmbeddingWrapper(vl_model_config, fp16_precision)
        assert wrapper.get_embedding_dim() == 2048
