"""Tests for MTEB dataset loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestMTEBDatasetLoader:
    """Tests for MTEB dataset loading."""

    def test_list_available_mteb_datasets(self) -> None:
        """Should list available MTEB retrieval datasets."""
        from embedding_tests.config.mteb_datasets import MTEB_RETRIEVAL_TASKS

        # CQADupStack should be included
        assert "cqadupstack-programmers" in MTEB_RETRIEVAL_TASKS
        assert "cqadupstack-unix" in MTEB_RETRIEVAL_TASKS
        assert "cqadupstack-tex" in MTEB_RETRIEVAL_TASKS

    def test_mteb_dataset_has_task_name(self) -> None:
        """Each MTEB dataset should have task name."""
        from embedding_tests.config.mteb_datasets import MTEB_RETRIEVAL_TASKS

        for name, task_name in MTEB_RETRIEVAL_TASKS.items():
            assert task_name, f"{name} missing task_name"
            # Task names should be PascalCase or contain Retrieval
            assert task_name[0].isupper(), f"{name} task should be PascalCase"

    def _create_mock_corpus(self, data: list[dict]) -> MagicMock:
        """Create a mock HuggingFace Dataset for corpus."""
        mock = MagicMock()
        mock.__iter__ = lambda self: iter(data)
        mock.column_names = ["id", "title", "text"]
        return mock

    def _create_mock_queries(self, data: list[dict]) -> MagicMock:
        """Create a mock HuggingFace Dataset for queries."""
        mock = MagicMock()
        mock.__iter__ = lambda self: iter(data)
        mock.column_names = ["id", "text"]
        return mock

    @patch("mteb.get_tasks")
    def test_load_mteb_dataset_converts_format(self, mock_get_tasks: MagicMock) -> None:
        """MTEB corpus should be converted to our format."""
        from embedding_tests.config.mteb_datasets import load_mteb_dataset

        # Mock MTEB task with new data structure
        mock_task = MagicMock()
        mock_task.dataset = {
            "default": {
                "test": {
                    "corpus": self._create_mock_corpus([
                        {"id": "doc1", "title": "Title 1", "text": "Text 1"},
                        {"id": "doc2", "title": "Title 2", "text": "Text 2"},
                    ]),
                    "queries": self._create_mock_queries([
                        {"id": "q1", "text": "Query 1"},
                    ]),
                    "relevant_docs": {
                        "q1": {"doc1": 1},
                    },
                }
            }
        }

        mock_get_tasks.return_value = [mock_task]

        corpus, queries = load_mteb_dataset("cqadupstack-programmers")

        assert len(corpus) == 2
        assert corpus[0]["doc_id"] == "doc1"
        assert "Text 1" in corpus[0]["text"]

    @patch("mteb.get_tasks")
    def test_load_mteb_queries_converts_format(self, mock_get_tasks: MagicMock) -> None:
        """MTEB queries should be converted to our format."""
        from embedding_tests.config.mteb_datasets import load_mteb_dataset

        mock_task = MagicMock()
        mock_task.dataset = {
            "default": {
                "test": {
                    "corpus": self._create_mock_corpus([]),
                    "queries": self._create_mock_queries([
                        {"id": "q1", "text": "Query 1"},
                        {"id": "q2", "text": "Query 2"},
                    ]),
                    "relevant_docs": {},
                }
            }
        }

        mock_get_tasks.return_value = [mock_task]

        corpus, queries = load_mteb_dataset("cqadupstack-programmers")

        assert len(queries) == 2
        assert queries[0]["query_id"] == "q1"
        assert queries[0]["text"] == "Query 1"

    @patch("mteb.get_tasks")
    def test_load_mteb_includes_relevance(self, mock_get_tasks: MagicMock) -> None:
        """MTEB queries should include relevant_doc_ids."""
        from embedding_tests.config.mteb_datasets import load_mteb_dataset

        mock_task = MagicMock()
        mock_task.dataset = {
            "default": {
                "test": {
                    "corpus": self._create_mock_corpus([
                        {"id": "doc1", "title": "", "text": "Text"},
                    ]),
                    "queries": self._create_mock_queries([
                        {"id": "q1", "text": "Query 1"},
                    ]),
                    "relevant_docs": {"q1": {"doc1": 1}},
                }
            }
        }

        mock_get_tasks.return_value = [mock_task]

        corpus, queries = load_mteb_dataset("cqadupstack-programmers")

        assert queries[0]["relevant_doc_ids"] == ["doc1"]

    def test_load_mteb_invalid_dataset_raises(self) -> None:
        """Should raise for unknown MTEB dataset."""
        from embedding_tests.config.mteb_datasets import load_mteb_dataset

        with pytest.raises(ValueError, match="Unknown MTEB retrieval task"):
            load_mteb_dataset("nonexistent_dataset")

    @patch("mteb.get_tasks")
    def test_load_mteb_with_limit(self, mock_get_tasks: MagicMock) -> None:
        """Should support limiting corpus and query count."""
        from embedding_tests.config.mteb_datasets import load_mteb_dataset

        mock_task = MagicMock()
        mock_task.dataset = {
            "default": {
                "test": {
                    "corpus": self._create_mock_corpus([
                        {"id": f"doc{i}", "title": "", "text": f"Text {i}"}
                        for i in range(100)
                    ]),
                    "queries": self._create_mock_queries([
                        {"id": f"q{i}", "text": f"Query {i}"}
                        for i in range(50)
                    ]),
                    "relevant_docs": {},
                }
            }
        }

        mock_get_tasks.return_value = [mock_task]

        corpus, queries = load_mteb_dataset(
            "cqadupstack-programmers", max_corpus=10, max_queries=5
        )

        assert len(corpus) == 10
        assert len(queries) == 5

    def test_list_mteb_datasets_function(self) -> None:
        """Should have a function to list datasets."""
        from embedding_tests.config.mteb_datasets import list_mteb_datasets

        datasets = list_mteb_datasets()
        assert len(datasets) >= 9  # At least CQADupStack suite
        assert all("name" in d and "task_name" in d for d in datasets)

    def test_is_mteb_dataset(self) -> None:
        """Should detect MTEB dataset names."""
        from embedding_tests.config.mteb_datasets import is_mteb_dataset

        assert is_mteb_dataset("cqadupstack-programmers")
        assert is_mteb_dataset("cqadupstack-unix")
        assert not is_mteb_dataset("programmers")
        assert not is_mteb_dataset("sample")

    @patch("mteb.get_tasks")
    def test_load_mteb_handles_dict_corpus_format(self, mock_get_tasks: MagicMock) -> None:
        """Should handle corpus with dict format (older MTEB format)."""
        from embedding_tests.config.mteb_datasets import load_mteb_dataset

        # Some MTEB tasks may use dict format for corpus/queries
        mock_corpus = {
            "doc1": {"text": "Text only", "title": ""},
            "doc2": "Just a string",
        }
        mock_queries = {
            "q1": "Query text",
        }

        mock_task = MagicMock()
        mock_task.dataset = {
            "default": {
                "test": {
                    "corpus": mock_corpus,
                    "queries": mock_queries,
                    "relevant_docs": {},
                }
            }
        }

        mock_get_tasks.return_value = [mock_task]

        corpus, queries = load_mteb_dataset("cqadupstack-programmers")

        assert len(corpus) == 2
        assert len(queries) == 1

    @patch("mteb.get_tasks")
    def test_load_mteb_raises_on_missing_split(self, mock_get_tasks: MagicMock) -> None:
        """Should raise when requested split is not available."""
        from embedding_tests.config.mteb_datasets import load_mteb_dataset

        mock_task = MagicMock()
        mock_task.dataset = {"default": {}}  # No test split

        mock_get_tasks.return_value = [mock_task]

        with pytest.raises(ValueError, match="No data found"):
            load_mteb_dataset("cqadupstack-programmers")
