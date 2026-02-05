"""Tests for MTEB benchmark integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embedding_tests.evaluation.mteb_runner import MTEBModelAdapter, run_mteb_tasks


class TestMTEBModelAdapter:
    """Tests for MTEB model adapter."""

    def test_mteb_model_adapter_wraps_encode(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])

        adapter = MTEBModelAdapter(mock_model)
        result = adapter.encode(["test text"])
        assert isinstance(result, np.ndarray)
        call_kwargs = mock_model.encode.call_args.kwargs
        assert call_kwargs.get("is_query") is False

    def test_mteb_model_adapter_handles_query_encoding(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])

        adapter = MTEBModelAdapter(mock_model)
        adapter.encode_queries(["test query"])
        # Should call encode with is_query=True
        call_kwargs = mock_model.encode.call_args.kwargs
        assert call_kwargs.get("is_query") is True

    def test_mteb_model_adapter_encodes_corpus(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        adapter = MTEBModelAdapter(mock_model)
        corpus = [{"text": "doc one"}, {"title": "doc two"}]
        adapter.encode_corpus(corpus)
        call_args = mock_model.encode.call_args
        assert call_args.args[0] == ["doc one", "doc two"]
        assert call_args.kwargs.get("is_query") is False


class TestRunMTEBTasks:
    """Tests for MTEB task runner."""

    def test_mteb_runner_returns_structured_results(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])

        # Don't actually run MTEB, just verify structure
        results = run_mteb_tasks(mock_model, task_types=[], dry_run=True)
        assert isinstance(results, dict)

    def test_mteb_runner_handles_import_error(self) -> None:
        mock_model = MagicMock()
        import builtins
        original_import = builtins.__import__

        def _mock_import(name: str, *args, **kwargs):
            if name == "mteb":
                raise ImportError("mocked mteb not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            results = run_mteb_tasks(mock_model, task_types=["Retrieval"], dry_run=False)
        assert "error" in results
        assert results["error"] == "mteb not installed"
