"""Tests for MTEB benchmark integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from embedding_tests.evaluation.mteb_runner import MTEBModelAdapter, run_mteb_tasks


class TestMTEBModelAdapter:
    """Tests for MTEB model adapter."""

    def test_mteb_model_adapter_wraps_encode(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_model.get_embedding_dim.return_value = 2

        adapter = MTEBModelAdapter(mock_model)
        result = adapter.encode(["test text"])
        assert isinstance(result, np.ndarray)

    def test_mteb_model_adapter_handles_query_encoding(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_model.get_embedding_dim.return_value = 2

        adapter = MTEBModelAdapter(mock_model)
        adapter.encode_queries(["test query"])
        # Should call encode with is_query=True
        call_kwargs = mock_model.encode.call_args.kwargs
        assert call_kwargs.get("is_query") is True


class TestRunMTEBTasks:
    """Tests for MTEB task runner."""

    def test_mteb_runner_returns_structured_results(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_model.get_embedding_dim.return_value = 2

        # Don't actually run MTEB, just verify structure
        results = run_mteb_tasks(mock_model, task_types=[], dry_run=True)
        assert isinstance(results, dict)
