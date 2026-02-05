"""Tests for vector retrieval with ChromaDB."""

from __future__ import annotations

import numpy as np
import pytest

from embedding_tests.pipeline.retrieval import VectorStore


@pytest.fixture(autouse=True)
def _cleanup_stores():
    """Ensure ChromaDB stores are cleaned up after each test."""
    yield
    # Reset the shared in-memory ChromaDB state so collection names don't leak
    import chromadb
    client = chromadb.Client()
    for col in client.list_collections():
        client.delete_collection(col.name)


class TestDistanceToScore:
    """Tests for _distance_to_score conversion."""

    def test_cosine_distance_zero_returns_one(self) -> None:
        store = VectorStore(collection_name="test_d2s_zero", embedding_dim=2, metric="cosine")
        assert store._distance_to_score(0.0) == 1.0

    def test_cosine_distance_two_returns_zero(self) -> None:
        store = VectorStore(collection_name="test_d2s_two", embedding_dim=2, metric="cosine")
        assert store._distance_to_score(2.0) == 0.0

    def test_cosine_distance_one_returns_half(self) -> None:
        store = VectorStore(collection_name="test_d2s_one", embedding_dim=2, metric="cosine")
        assert store._distance_to_score(1.0) == 0.5

    def test_l2_distance_zero_returns_one(self) -> None:
        store = VectorStore(collection_name="test_l2_zero", embedding_dim=2, metric="l2")
        assert store._distance_to_score(0.0) == 1.0

    def test_l2_distance_one_returns_half(self) -> None:
        store = VectorStore(collection_name="test_l2_one", embedding_dim=2, metric="l2")
        assert store._distance_to_score(1.0) == 0.5

    def test_ip_distance_returns_normalized_score(self) -> None:
        store = VectorStore(collection_name="test_ip", embedding_dim=2, metric="ip")
        # IP: dot = 1.0 - distance, score = clamp((dot + 1.0) / 2.0, 0, 1)
        # distance=1.0 -> dot=0.0 -> (0.0+1.0)/2.0 = 0.5
        assert store._distance_to_score(1.0) == 0.5
        # distance=-1.0 -> dot=2.0 -> (2.0+1.0)/2.0 = 1.5 -> clamped to 1.0
        assert store._distance_to_score(-1.0) == 1.0
        # distance=0.0 -> dot=1.0 -> (1.0+1.0)/2.0 = 1.0
        assert store._distance_to_score(0.0) == 1.0
        # distance=2.0 -> dot=-1.0 -> (-1.0+1.0)/2.0 = 0.0
        assert store._distance_to_score(2.0) == 0.0


class TestVectorStoreValidation:
    """Tests for VectorStore input validation."""

    def test_invalid_metric_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            VectorStore(collection_name="bad_metric", embedding_dim=3, metric="hamming")

    def test_query_embedding_dimension_mismatch_raises(self) -> None:
        store = VectorStore(collection_name="test_qdim", embedding_dim=3)
        embeddings = np.array([[1.0, 0.0, 0.0]])
        store.index(embeddings, ["d1"])
        wrong_dim_query = np.array([1.0, 0.0])  # 2D instead of 3D
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.query(wrong_dim_query, top_k=1)


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
        for r in results:
            assert 0.0 <= r.score <= 1.0

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

    def test_index_raises_on_count_mismatch(self) -> None:
        store = VectorStore(collection_name="test_mismatch", embedding_dim=3)
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="must match"):
            store.index(embeddings, ["d1"])

    def test_index_raises_on_dimension_mismatch(self) -> None:
        store = VectorStore(collection_name="test_dim", embedding_dim=3)
        embeddings = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.index(embeddings, ["d1"])
