"""GPU integration tests for NanoBEIR benchmark datasets.

Run with: pytest -m gpu tests/integration/test_nanobeir_benchmark.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch

from embedding_tests.config.datasets import load_dataset
from embedding_tests.config.models import PrecisionLevel, load_model_config
from embedding_tests.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from embedding_tests.hardware.precision import get_precision_config
from embedding_tests.models.loader import load_model
from embedding_tests.pipeline.rag import RagPipeline

if TYPE_CHECKING:
    from embedding_tests.config.hardware import GpuCapabilities


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


class TestNanoBEIRBenchmark:
    """Integration tests using NanoBEIR datasets."""

    @pytest.fixture
    def nanobeir_mock_data(self) -> tuple[list[dict], list[dict]]:
        """Create mock NanoBEIR-like data for testing."""
        corpus = [
            {"doc_id": f"doc{i}", "text": f"Document {i} contains information about topic {i % 5}."}
            for i in range(100)
        ]
        queries = [
            {
                "query_id": f"q{i}",
                "text": f"What is topic {i % 5}?",
                "relevant_doc_ids": [f"doc{j}" for j in range(i % 5, 100, 5)],
            }
            for i in range(10)
        ]
        return corpus, queries

    def test_nanobeir_pipeline_produces_meaningful_metrics(
        self, gpu: GpuCapabilities, configs_dir, nanobeir_mock_data
    ) -> None:
        """Run RAG pipeline on NanoBEIR-like data and verify metrics.

        This test validates that:
        1. The pipeline runs successfully with real embeddings
        2. Metrics are computed correctly
        3. Results are meaningful (not all zeros)
        """
        corpus, queries = nanobeir_mock_data

        embed_config = load_model_config(configs_dir / "models" / "qwen3_embedding_06b.yaml")
        precision = get_precision_config(gpu, PrecisionLevel.FP16)
        embed_model = load_model(embed_config, precision)

        try:
            pipeline = RagPipeline(
                embedding_model=embed_model,
                chunk_size=256,
                chunk_overlap=25,
                top_k=10,
            )
            result = pipeline.run(corpus, queries)

            assert result is not None
            assert len(result.query_results) == len(queries)

            # Compute metrics for each query
            recalls = []
            precisions = []
            mrr_inputs = []

            for qr in result.query_results:
                relevant = set(qr.relevant_doc_ids)
                retrieved = qr.retrieved_doc_ids

                r_k = recall_at_k(retrieved, relevant, k=10)
                p_k = precision_at_k(retrieved, relevant, k=10)

                recalls.append(r_k)
                precisions.append(p_k)
                mrr_inputs.append((retrieved, relevant))

            mrr_score = mrr(mrr_inputs)

            # Verify metrics are meaningful (not all zeros)
            avg_recall = sum(recalls) / len(recalls)
            avg_precision = sum(precisions) / len(precisions)

            # With mock data, we expect some retrieval success
            assert avg_recall > 0, "Recall should be > 0 with related documents"
            assert avg_precision > 0, "Precision should be > 0"
            assert mrr_score > 0, "MRR should be > 0"

            # Log results for debugging
            print(f"\nMetrics on mock NanoBEIR data:")
            print(f"  Avg Recall@10: {avg_recall:.3f}")
            print(f"  Avg Precision@10: {avg_precision:.3f}")
            print(f"  MRR: {mrr_score:.3f}")
            print(f"  Total time: {result.total_time_seconds:.2f}s")

        finally:
            embed_model.unload()
            torch.cuda.empty_cache()

    @patch("embedding_tests.config.nanobeir_datasets.hf_load_dataset")
    def test_load_nanobeir_dataset_structure(self, mock_hf_load) -> None:
        """Verify NanoBEIR dataset loading produces correct structure."""
        # Mock HuggingFace response
        mock_corpus = MagicMock()
        mock_corpus.__iter__ = lambda self: iter([
            {"_id": "doc1", "title": "Title", "text": "Content 1"},
            {"_id": "doc2", "title": "", "text": "Content 2"},
        ])
        mock_corpus.__len__ = lambda self: 2

        mock_queries = MagicMock()
        mock_queries.__iter__ = lambda self: iter([
            {"_id": "q1", "text": "Query 1"},
        ])
        mock_queries.__len__ = lambda self: 1

        mock_qrels = MagicMock()
        mock_qrels.__iter__ = lambda self: iter([
            {"query-id": "q1", "corpus-id": "doc1", "score": 1},
        ])

        mock_hf_load.return_value = {
            "corpus": mock_corpus,
            "queries": mock_queries,
            "qrels": mock_qrels,
        }

        corpus, queries = load_dataset("nano-nfcorpus")

        assert len(corpus) == 2
        assert len(queries) == 1
        assert queries[0]["relevant_doc_ids"] == ["doc1"]
        assert "Title" in corpus[0]["text"]  # Title should be included
        assert corpus[1]["text"] == "Content 2"  # Empty title doesn't add newlines


class TestMultiDatasetExperiment:
    """Integration tests for multi-dataset experiments."""

    def test_experiment_config_with_multiple_datasets(self, configs_dir) -> None:
        """Verify experiment config can specify multiple datasets."""
        from embedding_tests.config.experiment import load_experiment_config

        config = load_experiment_config(
            configs_dir / "experiments" / "quick_benchmark.yaml",
            configs_dir / "models",
        )

        assert len(config.datasets) >= 1
        assert "nano-nfcorpus" in config.datasets or "nano-scifact" in config.datasets

    def test_mteb_config_with_tasks(self, configs_dir) -> None:
        """Verify MTEB config can specify task names."""
        from embedding_tests.config.experiment import load_experiment_config

        config = load_experiment_config(
            configs_dir / "experiments" / "full_mteb.yaml",
            configs_dir / "models",
        )

        assert len(config.mteb_tasks) >= 1
        assert "NFCorpus" in config.mteb_tasks or "SciFact" in config.mteb_tasks
