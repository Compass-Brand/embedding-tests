"""Tests for vector retrieval with ChromaDB."""

from __future__ import annotations

import numpy as np
import pytest

from embedding_tests.pipeline.retrieval import VectorStore


class TestVectorStore:
    """Tests for VectorStore."""

    def test_index_documents_adds_to_store(self) -> None:
        store = VectorStore(collection_name="test", embedding_dim=3)
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        doc_ids = ["d1", "d2", "d3"]
        store.index(embeddings, doc_ids)
        assert store.count() == 3

    def test_query_returns_top_k_results(self) -> None:
        store = VectorStore(collection_name="test_topk", embedding_dim=3)
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        store.index(embeddings, ["d1", "d2", "d3"])

        query = np.array([1.0, 0.1, 0.0])
        results = store.query(query, top_k=2)
        assert len(results) == 2

    def test_query_results_sorted_by_score(self) -> None:
        store = VectorStore(collection_name="test_sorted", embedding_dim=3)
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        store.index(embeddings, ["d1", "d2", "d3"])

        query = np.array([1.0, 0.1, 0.0])
        results = store.query(query, top_k=3)
        # Results should be sorted by similarity (highest first)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_returns_document_ids_and_scores(self) -> None:
        store = VectorStore(collection_name="test_ids", embedding_dim=3)
        embeddings = np.array([[1.0, 0.0, 0.0]])
        store.index(embeddings, ["doc_abc"])

        query = np.array([1.0, 0.0, 0.0])
        results = store.query(query, top_k=1)
        assert results[0].doc_id == "doc_abc"
        assert isinstance(results[0].score, float)

    def test_cosine_similarity(self) -> None:
        store = VectorStore(collection_name="test_cosine", embedding_dim=2, metric="cosine")
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        store.index(embeddings, ["d1", "d2"])

        query = np.array([1.0, 0.0])
        results = store.query(query, top_k=2)
        assert results[0].doc_id == "d1"

    def test_clear_removes_all_documents(self) -> None:
        store = VectorStore(collection_name="test_clear", embedding_dim=3)
        embeddings = np.array([[1.0, 0.0, 0.0]])
        store.index(embeddings, ["d1"])
        assert store.count() == 1
        store.clear()
        assert store.count() == 0

    def test_query_empty_store_returns_empty_list(self) -> None:
        store = VectorStore(collection_name="test_empty", embedding_dim=3)
        results = store.query(np.array([1.0, 0.0, 0.0]), top_k=5)
        assert results == []
