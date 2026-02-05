"""Tests for MTEB benchmark integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


class TestMTEBModelAdapter:
    """Tests for the MTEB model adapter."""

    def test_adapter_encode_delegates_to_model(self) -> None:
        """Adapter encode should call model.encode with is_query=False."""
        from embedding_tests.evaluation.mteb_runner import MTEBModelAdapter

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1, 2, 3]])

        adapter = MTEBModelAdapter(mock_model)
        result = adapter.encode(["test sentence"])

        mock_model.encode.assert_called_once_with(
            ["test sentence"], is_query=False, batch_size=32
        )
        assert result.shape == (1, 3)

    def test_adapter_encode_queries_uses_query_mode(self) -> None:
        """Adapter encode_queries should call model.encode with is_query=True."""
        from embedding_tests.evaluation.mteb_runner import MTEBModelAdapter

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1, 2, 3]])

        adapter = MTEBModelAdapter(mock_model)
        adapter.encode_queries(["test query"])

        mock_model.encode.assert_called_once_with(
            ["test query"], is_query=True, batch_size=32
        )

    def test_adapter_encode_corpus_extracts_text(self) -> None:
        """Adapter encode_corpus should extract text from corpus dicts."""
        from embedding_tests.evaluation.mteb_runner import MTEBModelAdapter

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1, 2, 3], [4, 5, 6]])

        adapter = MTEBModelAdapter(mock_model)
        corpus = [
            {"text": "doc 1 text", "title": "Title 1"},
            {"text": "doc 2 text"},
        ]
        adapter.encode_corpus(corpus)

        call_args = mock_model.encode.call_args
        assert call_args[0][0] == ["doc 1 text", "doc 2 text"]

    def test_adapter_encode_corpus_falls_back_to_title(self) -> None:
        """Adapter encode_corpus should fall back to title if text is missing."""
        from embedding_tests.evaluation.mteb_runner import MTEBModelAdapter

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1, 2, 3]])

        adapter = MTEBModelAdapter(mock_model)
        corpus = [{"title": "Title only"}]
        adapter.encode_corpus(corpus)

        call_args = mock_model.encode.call_args
        assert call_args[0][0] == ["Title only"]

    def test_adapter_encode_corpus_handles_empty_doc(self) -> None:
        """Adapter encode_corpus should handle docs with no text or title."""
        from embedding_tests.evaluation.mteb_runner import MTEBModelAdapter

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1, 2, 3]])

        adapter = MTEBModelAdapter(mock_model)
        corpus = [{}]
        adapter.encode_corpus(corpus)

        call_args = mock_model.encode.call_args
        assert call_args[0][0] == [""]


class TestRunMTEBTasks:
    """Tests for run_mteb_tasks function."""

    def test_run_mteb_requires_task_selection(self) -> None:
        """Should raise if neither task_types nor task_names provided."""
        from embedding_tests.evaluation.mteb_runner import run_mteb_tasks

        mock_model = MagicMock()

        with pytest.raises(ValueError, match="Either task_types or task_names"):
            run_mteb_tasks(mock_model)

    def test_run_mteb_rejects_both_selectors(self) -> None:
        """Should raise if both task_types and task_names provided."""
        from embedding_tests.evaluation.mteb_runner import run_mteb_tasks

        mock_model = MagicMock()

        with pytest.raises(ValueError, match="only one of"):
            run_mteb_tasks(mock_model, task_types=["Retrieval"], task_names=["NFCorpus"])

    def test_run_mteb_rejects_empty_task_types(self) -> None:
        """Should raise if task_types is empty list."""
        from embedding_tests.evaluation.mteb_runner import run_mteb_tasks

        mock_model = MagicMock()

        with pytest.raises(ValueError, match="must not be empty"):
            run_mteb_tasks(mock_model, task_types=[])

    def test_run_mteb_rejects_empty_task_names(self) -> None:
        """Should raise if task_names is empty list."""
        from embedding_tests.evaluation.mteb_runner import run_mteb_tasks

        mock_model = MagicMock()

        with pytest.raises(ValueError, match="must not be empty"):
            run_mteb_tasks(mock_model, task_names=[])

    def test_run_mteb_dry_run_returns_empty(self) -> None:
        """Dry run should return empty results without executing."""
        from embedding_tests.evaluation.mteb_runner import run_mteb_tasks

        mock_model = MagicMock()

        result = run_mteb_tasks(mock_model, task_types=["Retrieval"], dry_run=True)

        assert result["dry_run"] is True
        assert result["tasks"] == []
        mock_model.encode.assert_not_called()

    def test_run_mteb_handles_import_error(self, monkeypatch) -> None:
        """Should return error dict when MTEB is not installed."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mteb":
                raise ImportError("No module named 'mteb'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        mock_model = MagicMock()

        from embedding_tests.evaluation import mteb_runner

        result = mteb_runner.run_mteb_tasks(mock_model, task_types=["Retrieval"])

        assert result == {"tasks": [], "error": "mteb not installed"}
        mock_model.encode.assert_not_called()


class TestMTEBTaskInfo:
    """Tests for MTEB task information."""

    def test_list_mteb_task_types(self) -> None:
        """Should list available MTEB task types."""
        from embedding_tests.evaluation.mteb_runner import MTEB_TASK_TYPES

        assert "Retrieval" in MTEB_TASK_TYPES
        assert "Reranking" in MTEB_TASK_TYPES
        assert "Clustering" in MTEB_TASK_TYPES

    def test_list_recommended_retrieval_tasks(self) -> None:
        """Should have recommended retrieval tasks for quick evaluation."""
        from embedding_tests.evaluation.mteb_runner import RECOMMENDED_RETRIEVAL_TASKS

        assert "NFCorpus" in RECOMMENDED_RETRIEVAL_TASKS
        assert "SciFact" in RECOMMENDED_RETRIEVAL_TASKS


class TestMTEBResultsFormatter:
    """Tests for MTEB results formatting."""

    def test_format_mteb_results_extracts_ndcg(self) -> None:
        """Should extract NDCG@10 from MTEB results."""
        from embedding_tests.evaluation.mteb_runner import format_mteb_results

        raw_results = [
            MagicMock(
                task_name="NFCorpus",
                scores={"test": [{"ndcg_at_10": 0.85, "mrr_at_10": 0.90}]},
            )
        ]

        formatted = format_mteb_results(raw_results)

        assert "NFCorpus" in formatted
        assert formatted["NFCorpus"]["ndcg_at_10"] == 0.85

    def test_format_mteb_results_handles_empty(self) -> None:
        """Should handle empty results gracefully."""
        from embedding_tests.evaluation.mteb_runner import format_mteb_results

        formatted = format_mteb_results([])

        assert formatted == {}

    def test_format_mteb_results_prefers_test_split(self) -> None:
        """Should prefer 'test' split when multiple splits exist."""
        from embedding_tests.evaluation.mteb_runner import format_mteb_results

        raw_results = [
            MagicMock(
                task_name="NFCorpus",
                scores={
                    "dev": [{"ndcg_at_10": 0.80}],
                    "test": [{"ndcg_at_10": 0.85}],
                },
            )
        ]

        formatted = format_mteb_results(raw_results)

        assert formatted["NFCorpus"]["ndcg_at_10"] == 0.85  # test split, not dev

    def test_format_mteb_results_falls_back_to_first_split(self) -> None:
        """Should fall back to first split if 'test' not present."""
        from embedding_tests.evaluation.mteb_runner import format_mteb_results

        raw_results = [
            MagicMock(
                task_name="NFCorpus",
                scores={"dev": [{"ndcg_at_10": 0.80}]},
            )
        ]

        formatted = format_mteb_results(raw_results)

        assert formatted["NFCorpus"]["ndcg_at_10"] == 0.80

    def test_format_mteb_results_filters_non_numeric(self) -> None:
        """Should filter out non-numeric metric values."""
        from embedding_tests.evaluation.mteb_runner import format_mteb_results

        raw_results = [
            MagicMock(
                task_name="NFCorpus",
                scores={"test": [{"ndcg_at_10": 0.85, "hf_subset": "default"}]},
            )
        ]

        formatted = format_mteb_results(raw_results)

        assert "ndcg_at_10" in formatted["NFCorpus"]
        assert "hf_subset" not in formatted["NFCorpus"]

    def test_format_mteb_results_handles_non_dict_scores(self) -> None:
        """Should skip non-dict score entries gracefully."""
        from embedding_tests.evaluation.mteb_runner import format_mteb_results

        raw_results = [
            MagicMock(
                task_name="NFCorpus",
                scores={"test": ["not a dict"]},  # Invalid structure
            )
        ]

        formatted = format_mteb_results(raw_results)

        # Should not crash, just return empty metrics for this task
        assert formatted == {}

    def test_format_mteb_results_does_not_fallback_for_empty_test_split(self) -> None:
        """Should not fall back to other splits when 'test' exists but is empty."""
        from embedding_tests.evaluation.mteb_runner import format_mteb_results

        raw_results = [
            MagicMock(
                task_name="NFCorpus",
                scores={"test": [], "dev": [{"ndcg_at_10": 0.80}]},
            )
        ]

        formatted = format_mteb_results(raw_results)

        # Empty test split means no metrics extracted, no fallback to dev
        assert formatted == {}

    def test_format_mteb_results_skips_malformed_entries(self) -> None:
        """Should skip entries missing task_name or scores attributes."""
        from embedding_tests.evaluation.mteb_runner import format_mteb_results

        raw_results = [
            MagicMock(spec=[]),  # No attributes at all
            MagicMock(
                task_name="NFCorpus",
                scores={"test": [{"ndcg_at_10": 0.85}]},
            ),
        ]

        formatted = format_mteb_results(raw_results)

        # Should only include the valid entry
        assert len(formatted) == 1
        assert "NFCorpus" in formatted
