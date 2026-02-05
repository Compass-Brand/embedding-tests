"""Tests for experiment runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embedding_tests.config.models import ModelConfig, ModelType, PrecisionLevel
from embedding_tests.runner.experiment import ExperimentRunner


@pytest.fixture
def model_configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            name="model-a",
            model_id="org/a",
            model_type=ModelType.TEXT_EMBEDDING,
            params_billions=0.6,
            embedding_dim=1024,
            supported_precisions=[PrecisionLevel.FP16, PrecisionLevel.INT8],
        ),
    ]


class TestExperimentRunner:
    """Tests for experiment runner."""

    @patch("embedding_tests.runner.experiment.load_model")
    @patch("embedding_tests.runner.experiment.get_precision_config")
    @patch("embedding_tests.runner.experiment.detect_gpu")
    def test_runner_iterates_model_precision_matrix(
        self,
        mock_detect: MagicMock,
        mock_precision: MagicMock,
        mock_loader: MagicMock,
        model_configs: list[ModelConfig],
        tmp_path: Path,
    ) -> None:
        from embedding_tests.config.hardware import GpuCapabilities
        mock_detect.return_value = GpuCapabilities(
            device_name="P40", compute_capability=(6, 1),
            total_vram_gb=24.0, supports_bf16=False, supports_flash_attn2=False,
        )
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda texts, **kw: np.random.randn(len(texts), 1024)
        mock_model.get_embedding_dim.return_value = 1024
        mock_loader.return_value = mock_model

        corpus = [{"doc_id": "d0", "text": "test doc"}]
        queries = [{"query_id": "q0", "text": "test", "relevant_doc_ids": ["d0"]}]

        runner = ExperimentRunner(
            model_configs=model_configs,
            precisions=[PrecisionLevel.FP16, PrecisionLevel.INT8],
            corpus=corpus,
            queries=queries,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        results = runner.run()
        # Should run 1 model * 2 precisions = 2 combinations
        assert len(results) == 2

    @patch("embedding_tests.runner.experiment.load_model")
    @patch("embedding_tests.runner.experiment.get_precision_config")
    @patch("embedding_tests.runner.experiment.detect_gpu")
    def test_runner_skips_completed_checkpoints(
        self,
        mock_detect: MagicMock,
        mock_precision: MagicMock,
        mock_loader: MagicMock,
        model_configs: list[ModelConfig],
        tmp_path: Path,
    ) -> None:
        from embedding_tests.config.hardware import GpuCapabilities
        from embedding_tests.runner.checkpoint import save_checkpoint

        mock_detect.return_value = GpuCapabilities(
            device_name="P40", compute_capability=(6, 1),
            total_vram_gb=24.0, supports_bf16=False, supports_flash_attn2=False,
        )
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda texts, **kw: np.random.randn(len(texts), 1024)
        mock_model.get_embedding_dim.return_value = 1024
        mock_loader.return_value = mock_model

        cp_dir = tmp_path / "checkpoints"
        # Pre-checkpoint one combination
        save_checkpoint(cp_dir, "model-a", "fp16", "completed", {"score": 0.5})

        runner = ExperimentRunner(
            model_configs=model_configs,
            precisions=[PrecisionLevel.FP16, PrecisionLevel.INT8],
            corpus=[{"doc_id": "d0", "text": "test"}],
            queries=[{"query_id": "q0", "text": "q", "relevant_doc_ids": ["d0"]}],
            checkpoint_dir=cp_dir,
        )
        results = runner.run()
        # FP16 was checkpointed, only INT8 should run fresh
        assert len(results) == 2  # Both results returned (one from checkpoint)
        assert mock_loader.call_count == 1

    @patch("embedding_tests.runner.experiment.load_model")
    @patch("embedding_tests.runner.experiment.get_precision_config")
    @patch("embedding_tests.runner.experiment.detect_gpu")
    def test_runner_handles_model_load_failure_gracefully(
        self,
        mock_detect: MagicMock,
        mock_precision: MagicMock,
        mock_loader: MagicMock,
        model_configs: list[ModelConfig],
        tmp_path: Path,
    ) -> None:
        from embedding_tests.config.hardware import GpuCapabilities

        mock_detect.return_value = GpuCapabilities(
            device_name="P40", compute_capability=(6, 1),
            total_vram_gb=24.0, supports_bf16=False, supports_flash_attn2=False,
        )
        mock_loader.side_effect = RuntimeError("CUDA OOM")

        runner = ExperimentRunner(
            model_configs=model_configs,
            precisions=[PrecisionLevel.FP16],
            corpus=[{"doc_id": "d0", "text": "test"}],
            queries=[{"query_id": "q0", "text": "q", "relevant_doc_ids": ["d0"]}],
            checkpoint_dir=tmp_path / "checkpoints",
        )
        # Should not raise, just log error
        results = runner.run()
        assert len(results) == 1
        assert results[0].get("error") is not None
