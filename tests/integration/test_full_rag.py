"""GPU integration tests for full RAG pipeline.

Run with: pytest -m gpu tests/integration/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from embedding_tests.config.models import PrecisionLevel, load_model_config
from embedding_tests.hardware.precision import get_precision_config
from embedding_tests.models.loader import load_model
from embedding_tests.pipeline.rag import RagPipeline

if TYPE_CHECKING:
    from embedding_tests.config.hardware import GpuCapabilities


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


class TestFullRAG:
    """Integration tests for complete RAG pipeline."""

    def test_full_rag_pipeline_with_small_models(
        self, gpu: GpuCapabilities, configs_dir, sample_corpus, sample_queries
    ) -> None:
        """Run complete RAG pipeline with Qwen3-Embedding-0.6B."""

        embed_config = load_model_config(configs_dir / "models" / "qwen3_embedding_06b.yaml")
        precision = get_precision_config(gpu, PrecisionLevel.FP16)
        embed_model = load_model(embed_config, precision)

        try:
            pipeline = RagPipeline(
                embedding_model=embed_model,
                chunk_size=200,
                chunk_overlap=20,
                top_k=5,
            )
            result = pipeline.run(sample_corpus, sample_queries)

            assert result is not None
            assert len(result.query_results) == len(sample_queries)
            assert result.total_time_seconds > 0
            assert result.num_corpus_chunks > 0
            assert result.used_reranker is False

            # Verify each query got results
            for qr in result.query_results:
                assert len(qr.retrieved_doc_ids) > 0

        finally:
            embed_model.unload()
            torch.cuda.empty_cache()
