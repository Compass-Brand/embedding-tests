"""Tests for RAG pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embedding_tests.pipeline.rag import RagPipeline, RagResult


class TestRagPipeline:
    """Tests for RAG pipeline orchestrator."""

    def _make_mock_model(self, dim: int = 4) -> MagicMock:
        mock = MagicMock()
        mock.get_embedding_dim.return_value = dim

        def _encode(texts, **kwargs):
            return np.random.randn(len(texts), dim).astype(np.float32)

        mock.encode.side_effect = _encode
        return mock

    def test_rag_pipeline_runs_full_chain(self) -> None:
        model = self._make_mock_model()
        corpus = [{"doc_id": f"d{i}", "text": f"document {i}"} for i in range(5)]
        queries = [{"query_id": "q1", "text": "test query", "relevant_doc_ids": ["d0"]}]

        pipeline = RagPipeline(embedding_model=model, chunk_size=100, chunk_overlap=10, top_k=3)
        result = pipeline.run(corpus, queries)
        assert isinstance(result, RagResult)
        assert len(result.query_results) == 1

    def test_rag_pipeline_with_reranking(self) -> None:
        model = self._make_mock_model()
        reranker = MagicMock()
        reranker.rerank.return_value = [(0, 0.9), (1, 0.5)]

        corpus = [{"doc_id": f"d{i}", "text": f"document {i}"} for i in range(5)]
        queries = [{"query_id": "q1", "text": "query", "relevant_doc_ids": ["d0"]}]

        pipeline = RagPipeline(
            embedding_model=model, reranker_model=reranker,
            chunk_size=100, chunk_overlap=10, top_k=3
        )
        result = pipeline.run(corpus, queries)
        assert result is not None

    def test_rag_pipeline_without_reranking(self) -> None:
        model = self._make_mock_model()
        corpus = [{"doc_id": "d0", "text": "content"}]
        queries = [{"query_id": "q1", "text": "query", "relevant_doc_ids": ["d0"]}]

        pipeline = RagPipeline(embedding_model=model, chunk_size=100, chunk_overlap=10, top_k=1)
        result = pipeline.run(corpus, queries)
        assert result is not None
        assert result.used_reranker is False

    def test_rag_pipeline_returns_results_with_metrics(self) -> None:
        model = self._make_mock_model()
        corpus = [{"doc_id": "d0", "text": "relevant content about AI"}]
        queries = [{"query_id": "q1", "text": "AI query", "relevant_doc_ids": ["d0"]}]

        pipeline = RagPipeline(embedding_model=model, chunk_size=100, chunk_overlap=10, top_k=1)
        result = pipeline.run(corpus, queries)
        assert result.total_time_seconds > 0
        assert len(result.query_results) > 0
