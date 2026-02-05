"""Tests for batch embedding pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from embedding_tests.pipeline.embedding import batch_embed


class TestBatchEmbed:
    """Tests for batch embedding."""

    def test_batch_embed_calls_model_encode(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.get_embedding_dim.return_value = 2

        result = batch_embed(mock_model, ["hello", "world"], batch_size=32)
        mock_model.encode.assert_called()
        assert result.embeddings.shape == (2, 2)

    def test_batch_embed_respects_batch_size(self) -> None:
        mock_model = MagicMock()
        # Return different arrays for each batch call
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2]] * 2),
            np.array([[0.3, 0.4]] * 2),
            np.array([[0.5, 0.6]]),
        ]
        mock_model.get_embedding_dim.return_value = 2

        result = batch_embed(mock_model, ["a", "b", "c", "d", "e"], batch_size=2)
        assert mock_model.encode.call_count == 3
        assert result.embeddings.shape == (5, 2)

    def test_batch_embed_records_latency(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_model.get_embedding_dim.return_value = 2

        result = batch_embed(mock_model, ["test"], batch_size=32)
        assert result.total_time_seconds > 0

    def test_batch_embed_returns_correct_shape(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(10, 768)
        mock_model.get_embedding_dim.return_value = 768

        result = batch_embed(mock_model, [f"text_{i}" for i in range(10)], batch_size=32)
        assert result.embeddings.shape == (10, 768)
