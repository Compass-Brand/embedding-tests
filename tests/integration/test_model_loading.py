"""GPU integration tests for model loading.

These tests require a CUDA GPU and will download model weights.
Run with: pytest -m gpu tests/integration/
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embedding_tests.config.hardware import detect_gpu
from embedding_tests.config.models import ModelType, PrecisionLevel, load_model_config
from embedding_tests.hardware.precision import get_precision_config
from embedding_tests.models.loader import load_model


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def gpu():
    """Detect GPU capabilities."""
    caps = detect_gpu()
    if caps is None:
        pytest.skip("No CUDA GPU available")
    return caps


class TestModelLoading:
    """Integration tests for loading real models on GPU."""

    def test_load_qwen3_embedding_06b_fp16(self, gpu, configs_dir) -> None:
        """Load smallest text embedding model and verify output."""
        config = load_model_config(configs_dir / "models" / "qwen3_embedding_06b.yaml")
        precision = get_precision_config(gpu, PrecisionLevel.FP16)

        model = load_model(config, precision)
        try:
            result = model.encode(["Hello, world!"])
            assert isinstance(result, np.ndarray)
            assert result.shape[1] == config.embedding_dim
            assert result.shape[0] == 1
        finally:
            model.unload()
            torch.cuda.empty_cache()

    def test_load_qwen3_vl_embedding_2b_fp16(self, gpu, configs_dir) -> None:
        """Load smallest VL embedding model and verify text encoding."""
        config = load_model_config(configs_dir / "models" / "qwen3_vl_embedding_2b.yaml")
        precision = get_precision_config(gpu, PrecisionLevel.FP16)

        model = load_model(config, precision)
        try:
            result = model.encode(["Test text for VL embedding"])
            assert isinstance(result, np.ndarray)
            assert result.shape[1] == config.embedding_dim
        finally:
            model.unload()
            torch.cuda.empty_cache()

    def test_load_qwen3_vl_reranker_2b_fp16(self, gpu, configs_dir) -> None:
        """Load smallest reranker model and verify reranking output."""
        config = load_model_config(configs_dir / "models" / "qwen3_vl_reranker_2b.yaml")
        precision = get_precision_config(gpu, PrecisionLevel.FP16)

        model = load_model(config, precision)
        try:
            results = model.rerank(
                "What is machine learning?",
                [
                    "Machine learning is a branch of AI.",
                    "The weather is sunny today.",
                    "Deep learning uses neural networks.",
                ],
                top_k=2,
            )
            assert len(results) == 2
            assert all(isinstance(r, tuple) for r in results)
            # ML-related docs should score higher
            scores = [r[1] for r in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            model.unload()
            torch.cuda.empty_cache()
