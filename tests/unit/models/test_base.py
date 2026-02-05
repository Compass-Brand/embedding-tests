"""Tests for model protocols."""

from __future__ import annotations


import numpy as np

from embedding_tests.models.base import EmbeddingModel, RerankerModel


class MockEmbeddingModel:
    """Mock implementation satisfying EmbeddingModel protocol."""

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        return np.zeros((len(texts), 768))

    def get_embedding_dim(self) -> int:
        return 768

    def unload(self) -> None:
        pass


class MockRerankerModel:
    """Mock implementation satisfying RerankerModel protocol."""

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        return [(0, 0.9), (1, 0.5)]

    def unload(self) -> None:
        pass


class TestEmbeddingModelProtocol:
    """Tests for EmbeddingModel protocol."""

    def test_mock_embedding_model_encode_returns_correct_shape(self) -> None:
        model = MockEmbeddingModel()
        result = model.encode(["hello"])
        assert result.shape == (1, 768)

    def test_mock_embedding_model_returns_embedding_dim(self) -> None:
        model = MockEmbeddingModel()
        assert model.get_embedding_dim() == 768

    def test_mock_model_satisfies_embedding_protocol(self) -> None:
        model = MockEmbeddingModel()
        assert isinstance(model, EmbeddingModel)


class TestRerankerModelProtocol:
    """Tests for RerankerModel protocol."""

    def test_mock_reranker_model_rerank_returns_results(self) -> None:
        model = MockRerankerModel()
        result = model.rerank("query", ["doc1", "doc2"])
        assert len(result) == 2
        assert result[0] == (0, 0.9)

    def test_mock_model_satisfies_reranker_protocol(self) -> None:
        model = MockRerankerModel()
        assert isinstance(model, RerankerModel)
