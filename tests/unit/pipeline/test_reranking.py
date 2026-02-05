"""Tests for two-stage reranking pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from embedding_tests.pipeline.reranking import rerank_results, RerankResult


class TestReranking:
    """Tests for reranking pipeline."""

    def test_rerank_reorders_retrieved_documents(self) -> None:
        mock_reranker = MagicMock()
        # Reranker reverses the order
        mock_reranker.rerank.return_value = [(2, 0.9), (0, 0.7), (1, 0.3)]

        docs = [
            {"doc_id": "d1", "text": "first doc"},
            {"doc_id": "d2", "text": "second doc"},
            {"doc_id": "d3", "text": "third doc"},
        ]
        results = rerank_results("test query", docs, mock_reranker, top_k=3)
        assert results[0].doc_id == "d3"
        assert results[1].doc_id == "d1"

    def test_rerank_uses_reranker_model_scores(self) -> None:
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [(0, 0.95)]

        docs = [{"doc_id": "d1", "text": "a doc"}]
        results = rerank_results("query", docs, mock_reranker, top_k=1)
        mock_reranker.rerank.assert_called_once()
        assert results[0].score == pytest.approx(0.95)

    def test_rerank_returns_top_k_from_larger_candidate_set(self) -> None:
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [(0, 0.9), (3, 0.8)]

        docs = [{"doc_id": f"d{i}", "text": f"doc {i}"} for i in range(10)]
        results = rerank_results("query", docs, mock_reranker, top_k=2)
        assert len(results) == 2

    def test_rerank_preserves_document_metadata(self) -> None:
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [(0, 0.9)]

        docs = [{"doc_id": "d1", "text": "content", "source": "wiki"}]
        results = rerank_results("query", docs, mock_reranker, top_k=1)
        assert results[0].doc_id == "d1"
        assert results[0].metadata.get("source") == "wiki"

    def test_rerank_raises_on_missing_doc_id(self) -> None:
        mock_reranker = MagicMock()
        docs = [{"text": "content"}]  # missing doc_id
        with pytest.raises(ValueError, match="missing required"):
            rerank_results("query", docs, mock_reranker, top_k=1)
        mock_reranker.rerank.assert_not_called()

    def test_rerank_raises_on_missing_text(self) -> None:
        mock_reranker = MagicMock()
        docs = [{"doc_id": "d1"}]  # missing text
        with pytest.raises(ValueError, match="missing required"):
            rerank_results("query", docs, mock_reranker, top_k=1)
        mock_reranker.rerank.assert_not_called()

    def test_rerank_handles_empty_documents(self) -> None:
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []
        results = rerank_results("query", [], mock_reranker, top_k=1)
        assert results == []

    def test_rerank_raises_on_invalid_reranker_index(self) -> None:
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [(5, 0.9)]  # out of bounds
        docs = [{"doc_id": "d1", "text": "content"}]
        with pytest.raises((ValueError, IndexError)):
            rerank_results("query", docs, mock_reranker, top_k=1)
