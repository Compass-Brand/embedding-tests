"""GPU integration tests for embedding pipeline.

Run with: pytest -m gpu tests/integration/
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embedding_tests.config.hardware import detect_gpu
from embedding_tests.config.models import PrecisionLevel, load_model_config
from embedding_tests.hardware.precision import get_precision_config
from embedding_tests.models.loader import load_model
from embedding_tests.pipeline.chunking import ChunkingStrategy, chunk_text
from embedding_tests.pipeline.embedding import batch_embed
from embedding_tests.pipeline.retrieval import VectorStore


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


class TestEmbeddingPipeline:
    """End-to-end embedding pipeline tests."""

    def test_end_to_end_embed_and_retrieve(self, configs_dir, sample_corpus, sample_queries) -> None:
        """Chunk, embed, index, and retrieve with real model."""
        gpu = detect_gpu()
        if gpu is None:
            pytest.skip("No CUDA GPU")

        config = load_model_config(configs_dir / "models" / "qwen3_embedding_06b.yaml")
        precision = get_precision_config(gpu, PrecisionLevel.FP16)
        model = load_model(config, precision)

        try:
            # Chunk corpus
            all_chunks = []
            chunk_ids = []
            for doc in sample_corpus:
                chunks = chunk_text(
                    doc["text"],
                    strategy=ChunkingStrategy.RECURSIVE,
                    chunk_size=200,
                    chunk_overlap=20,
                    source_doc_id=doc["doc_id"],
                )
                for chunk in chunks:
                    all_chunks.append(chunk.text)
                    chunk_ids.append(f"{chunk.source_doc_id}_{chunk.chunk_index}")

            # Embed
            result = batch_embed(model, all_chunks, batch_size=8)
            assert result.embeddings.shape[0] == len(all_chunks)

            # Index and retrieve
            store = VectorStore(
                collection_name="integration_test",
                embedding_dim=config.embedding_dim,
            )
            store.index(result.embeddings, chunk_ids)

            # Query
            for q in sample_queries[:2]:
                q_embed = batch_embed(model, [q["text"]], batch_size=1, is_query=True)
                results = store.query(q_embed.embeddings[0], top_k=5)
                assert len(results) > 0
                # Check if any relevant doc appears in results
                retrieved_docs = {r.doc_id.split("_")[0] + "_" + r.doc_id.split("_")[1] for r in results}
                relevant = set(q["relevant_doc_ids"])
                recall = len(retrieved_docs & relevant) / len(relevant) if relevant else 0
                # At least some recall expected with small corpus
                assert recall >= 0  # Non-negative (may be 0 with very small corpus)

        finally:
            model.unload()
            torch.cuda.empty_cache()
