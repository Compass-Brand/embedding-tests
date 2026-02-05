"""Tests for VL reranker wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

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


def _mock_lm(hidden_dim: int = 16, vocab_size: int = 100) -> MagicMock:
    """Create a mock Qwen3VLForConditionalGeneration with required internals."""
    mock_lm = MagicMock()
    mock_inner = MagicMock()
    mock_lm.model = mock_inner
    mock_inner.eval.return_value = None
    mock_inner.dtype = torch.float16

    cpu_param = torch.zeros(1, device="cpu")
    mock_inner.parameters = lambda: iter([cpu_param])

    # Real tensor so score_linear creation works
    mock_lm.lm_head.weight.data = torch.randn(vocab_size, hidden_dim)
    return mock_lm


def _mock_tokenizer() -> MagicMock:
    """Create a mock AutoTokenizer with vocab."""
    mock_tok = MagicMock()
    mock_tok.get_vocab.return_value = {"yes": 1, "no": 0}
    return mock_tok


class TestVLRerankerWrapper:
    """Tests for VLRerankerWrapper."""

    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch(
        "embedding_tests.models.vl_reranker_wrapper.Qwen3VLForConditionalGeneration"
    )
    def test_vl_reranker_init_uses_fp16(
        self,
        mock_gen_cls: MagicMock,
        mock_tok_cls: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        mock_gen_cls.from_pretrained.return_value = _mock_lm()
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()

        VLRerankerWrapper(reranker_config, fp16_precision)
        call_kwargs = mock_gen_cls.from_pretrained.call_args.kwargs
        assert call_kwargs.get("torch_dtype") == torch.float16

    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch(
        "embedding_tests.models.vl_reranker_wrapper.Qwen3VLForConditionalGeneration"
    )
    def test_vl_reranker_rerank_returns_sorted_index_score_tuples(
        self,
        mock_gen_cls: MagicMock,
        mock_tok_cls: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        mock_gen_cls.from_pretrained.return_value = _mock_lm()
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()

        wrapper = VLRerankerWrapper(reranker_config, fp16_precision)

        # Mock _score_single to return controlled per-document scores
        with patch.object(wrapper, "_score_single", side_effect=[0.9, 0.3, 0.7]):
            results = wrapper.rerank("query", ["doc1", "doc2", "doc3"])

        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Scores sorted descending
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)
        # Original: doc0=0.9, doc1=0.3, doc2=0.7 → sorted order: 0, 2, 1
        indices = [r[0] for r in results]
        assert indices == [0, 2, 1]

    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch(
        "embedding_tests.models.vl_reranker_wrapper.Qwen3VLForConditionalGeneration"
    )
    def test_vl_reranker_rerank_respects_top_k(
        self,
        mock_gen_cls: MagicMock,
        mock_tok_cls: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        mock_gen_cls.from_pretrained.return_value = _mock_lm()
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()

        wrapper = VLRerankerWrapper(reranker_config, fp16_precision)

        with patch.object(
            wrapper, "_score_single", side_effect=[0.9, 0.3, 0.7, 0.5, 0.1]
        ):
            results = wrapper.rerank("query", ["a", "b", "c", "d", "e"], top_k=2)

        assert len(results) == 2
        scores = [r[1] for r in results]
        assert scores[0] >= scores[1]

    @patch("embedding_tests.models.vl_reranker_wrapper.torch.cuda.empty_cache")
    @patch(
        "embedding_tests.models.vl_reranker_wrapper.torch.cuda.is_available",
        return_value=True,
    )
    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch(
        "embedding_tests.models.vl_reranker_wrapper.Qwen3VLForConditionalGeneration"
    )
    def test_vl_reranker_unload_clears_memory(
        self,
        mock_gen_cls: MagicMock,
        mock_tok_cls: MagicMock,
        mock_is_available: MagicMock,
        mock_empty_cache: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        mock_gen_cls.from_pretrained.return_value = _mock_lm()
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()

        wrapper = VLRerankerWrapper(reranker_config, fp16_precision)
        wrapper.unload()
        mock_empty_cache.assert_called_once()
        assert wrapper._model is None
        assert wrapper._score_linear is None
        assert wrapper._tokenizer is None

    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch(
        "embedding_tests.models.vl_reranker_wrapper.Qwen3VLForConditionalGeneration"
    )
    def test_vl_reranker_satisfies_reranker_protocol(
        self,
        mock_gen_cls: MagicMock,
        mock_tok_cls: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        mock_gen_cls.from_pretrained.return_value = _mock_lm()
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()

        wrapper = VLRerankerWrapper(reranker_config, fp16_precision)
        assert isinstance(wrapper, RerankerModel)

    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch(
        "embedding_tests.models.vl_reranker_wrapper.Qwen3VLForConditionalGeneration"
    )
    def test_vl_reranker_rerank_handles_empty_documents(
        self,
        mock_gen_cls: MagicMock,
        mock_tok_cls: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        mock_gen_cls.from_pretrained.return_value = _mock_lm()
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()

        wrapper = VLRerankerWrapper(reranker_config, fp16_precision)
        results = wrapper.rerank("query", [])
        assert results == []

    @patch("embedding_tests.models.vl_reranker_wrapper.AutoTokenizer")
    @patch(
        "embedding_tests.models.vl_reranker_wrapper.Qwen3VLForConditionalGeneration"
    )
    def test_vl_reranker_rerank_rejects_non_positive_top_k(
        self,
        mock_gen_cls: MagicMock,
        mock_tok_cls: MagicMock,
        reranker_config: ModelConfig,
        fp16_precision: PrecisionConfig,
    ) -> None:
        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        mock_gen_cls.from_pretrained.return_value = _mock_lm()
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()

        wrapper = VLRerankerWrapper(reranker_config, fp16_precision)
        with pytest.raises(ValueError, match="top_k must be positive"):
            wrapper.rerank("query", ["doc1"], top_k=0)

    def test_build_messages_format(self) -> None:
        from embedding_tests.models.vl_reranker_wrapper import VLRerankerWrapper

        messages = VLRerankerWrapper._build_messages(
            "test instruction", "test query", "test document"
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        user_content = messages[1]["content"]
        assert any("<Instruct>:" in item["text"] for item in user_content)
        assert any("<Query>:" in item["text"] for item in user_content)
        assert any("test query" == item["text"] for item in user_content)
        assert any("test document" == item["text"] for item in user_content)
