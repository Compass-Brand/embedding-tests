"""Tests for VL reranker wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from embedding_tests.config.models import ModelConfig, ModelType, PrecisionLevel
from embedding_tests.hardware.precision import PrecisionConfig
from embedding_tests.models.base import RerankerModel


@pytest.fixture
def reranker_config() -> ModelConfig:
    return ModelConfig(
        name="qwen3-vl-reranker-2b",
        model_id="Qwen/Qwen3-VL-Reranker-2B",
        model_type=ModelType.MULTIMODAL_RERANKER,
        params_billions=2.0,
        embedding_dim=0,
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


class TestVLRerankerWrapper:
    """Tests for VLRerankerWrapper."""

    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch("embedding_tests.models.vl_reranker_wrapper.AutoModelForSequenceClassification")
    def test_vl_reranker_init_uses_fp16(
        self,
        mock_auto_model: MagicMock,
        mock_tokenizer: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        VLRerankerWrapper(reranker_config, fp16_precision)
        call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
        assert "float16" in str(call_kwargs.get("torch_dtype", ""))

    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch("embedding_tests.models.vl_reranker_wrapper.AutoModelForSequenceClassification")
    def test_vl_reranker_rerank_returns_sorted_index_score_tuples(
        self,
        mock_auto_model: MagicMock,
        mock_tokenizer: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        import torch

        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        # Each call returns a single logit for one query-doc pair
        outputs = []
        for score in [0.9, 0.3, 0.7]:
            out = MagicMock()
            out.logits = torch.tensor([[score]])
            outputs.append(out)
        mock_model.side_effect = outputs
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        mock_tokenizer.from_pretrained.return_value = mock_tok

        wrapper = VLRerankerWrapper(reranker_config, fp16_precision)
        results = wrapper.rerank("query", ["doc1", "doc2", "doc3"])
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Scores should be sorted descending
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch("embedding_tests.models.vl_reranker_wrapper.AutoModelForSequenceClassification")
    def test_vl_reranker_rerank_respects_top_k(
        self,
        mock_auto_model: MagicMock,
        mock_tokenizer: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        import torch

        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        outputs = []
        for score in [0.9, 0.3, 0.7, 0.5, 0.1]:
            out = MagicMock()
            out.logits = torch.tensor([[score]])
            outputs.append(out)
        mock_model.side_effect = outputs
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        mock_tokenizer.from_pretrained.return_value = mock_tok

        wrapper = VLRerankerWrapper(reranker_config, fp16_precision)
        results = wrapper.rerank("query", ["a", "b", "c", "d", "e"], top_k=2)
        assert len(results) == 2

    @patch("embedding_tests.models.vl_reranker_wrapper.torch")
    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch("embedding_tests.models.vl_reranker_wrapper.AutoModelForSequenceClassification")
    def test_vl_reranker_unload_clears_memory(
        self,
        mock_auto_model: MagicMock,
        mock_tokenizer: MagicMock,
        mock_torch: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        wrapper = VLRerankerWrapper(reranker_config, fp16_precision)
        wrapper.unload()
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch("embedding_tests.models.vl_reranker_wrapper.AutoModelForSequenceClassification")
    def test_vl_reranker_satisfies_reranker_protocol(
        self,
        mock_auto_model: MagicMock,
        mock_tokenizer: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        wrapper = VLRerankerWrapper(reranker_config, fp16_precision)
        assert isinstance(wrapper, RerankerModel)
