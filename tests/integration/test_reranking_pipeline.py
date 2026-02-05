"""GPU integration tests for reranking pipeline.

Run with: pytest -m gpu tests/integration/
"""

from __future__ import annotations

import pytest
import torch

from embedding_tests.config.hardware import GpuCapabilities
from embedding_tests.config.models import PrecisionLevel, load_model_config
from embedding_tests.hardware.precision import get_precision_config
from embedding_tests.models.loader import load_model


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


class TestRerankingPipeline:
    """Integration tests for reranking pipeline."""

    def test_reranking_improves_retrieval_quality(self, gpu: GpuCapabilities, configs_dir) -> None:
        """Verify reranker properly scores query-document relevance."""

        config = load_model_config(configs_dir / "models" / "qwen3_vl_reranker_2b.yaml")
        precision = get_precision_config(gpu, PrecisionLevel.FP16)
        reranker = load_model(config, precision)

        try:
            query = "What are embedding models used for?"
            documents = [
                "The stock market crashed yesterday.",
                "Embedding models convert text to dense vectors for semantic search.",
                "Pizza is a popular food worldwide.",
                "Vector representations enable similarity matching in NLP.",
                "The cat sat on the mat.",
            ]

            results = reranker.rerank(query, documents, top_k=3)
            assert len(results) == 3

            # The relevant docs (indices 1, 3) should rank higher
            top_indices = {r[0] for r in results}
            assert 1 in top_indices and 3 in top_indices, f"Expected both 1 and 3 in {top_indices}"

        finally:
            reranker.unload()
            torch.cuda.empty_cache()
