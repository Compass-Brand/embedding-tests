"""Tests for CoIR (CodeSearchNet) dataset loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCoIRDatasetLoader:
    """Tests for CoIR dataset loading from HuggingFace."""

    def test_list_available_coir_datasets(self) -> None:
        """Should list available CoIR datasets."""
        from embedding_tests.config.coir_datasets import COIR_DATASETS

        expected_languages = {"python", "java", "javascript", "go", "php", "ruby"}
        expected_names = {f"codesearchnet-{lang}" for lang in expected_languages}
        assert set(COIR_DATASETS.keys()) == expected_names

    def test_coir_dataset_info_has_required_fields(self) -> None:
        """Each CoIR dataset should have metadata."""
        from embedding_tests.config.coir_datasets import COIR_DATASETS

        for name, info in COIR_DATASETS.items():
            assert "hf_name" in info, f"{name} missing hf_name"
            assert "language" in info, f"{name} missing language"

    @patch("embedding_tests.config.coir_datasets.hf_load_dataset")
    def test_load_coir_corpus_converts_format(self, mock_load: MagicMock) -> None:
        """CoIR corpus should be converted to our format."""
        from embedding_tests.config.coir_datasets import load_coir_dataset

        # Mock corpus, queries, and qrels datasets
        mock_corpus = MagicMock()
        mock_corpus.__iter__ = lambda self: iter([
            {"_id": "func1", "text": "def hello():\n    return 'world'"},
            {"_id": "func2", "text": "def add(a, b):\n    return a + b"},
        ])
        mock_corpus.__len__ = lambda self: 2

        mock_queries = MagicMock()
        mock_queries.__iter__ = lambda self: iter([
            {"_id": "q1", "text": "function that returns world"},
        ])
        mock_queries.__len__ = lambda self: 1

        # For CoIR qrels, each row has query_id, positive_passages, negative_passages
        mock_qrels = MagicMock()
        mock_qrels.__iter__ = lambda self: iter([
            {"query_id": "q1", "positive_passages": [{"docid": "func1"}], "negative_passages": []},
        ])

        def side_effect(name, subset=None, **kwargs):
            if subset and "corpus" in subset:
                return {"train": mock_corpus}
            elif subset and "queries" in subset:
                return {"train": mock_queries}
            elif subset and "qrels" in subset:
                return {"train": mock_qrels}
            return {}

        mock_load.side_effect = side_effect

        corpus, queries = load_coir_dataset("codesearchnet-python")

        assert len(corpus) == 2
        assert corpus[0]["doc_id"] == "func1"
        assert "def hello" in corpus[0]["text"]

    @patch("embedding_tests.config.coir_datasets.hf_load_dataset")
    def test_load_coir_queries_converts_format(self, mock_load: MagicMock) -> None:
        """CoIR queries should be converted to our format."""
        from embedding_tests.config.coir_datasets import load_coir_dataset

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = lambda self: iter([])
        mock_corpus.__len__ = lambda self: 0

        mock_queries = MagicMock()
        mock_queries.__iter__ = lambda self: iter([
            {"_id": "q1", "text": "Query 1"},
            {"_id": "q2", "text": "Query 2"},
        ])
        mock_queries.__len__ = lambda self: 2

        mock_qrels = MagicMock()
        mock_qrels.__iter__ = lambda self: iter([])

        def side_effect(name, subset=None, **kwargs):
            if subset and "corpus" in subset:
                return {"train": mock_corpus}
            elif subset and "queries" in subset:
                return {"train": mock_queries}
            elif subset and "qrels" in subset:
                return {"train": mock_qrels}
            return {}

        mock_load.side_effect = side_effect

        corpus, queries = load_coir_dataset("codesearchnet-python")

        assert len(queries) == 2
        assert queries[0]["query_id"] == "q1"
        assert queries[0]["text"] == "Query 1"

    @patch("embedding_tests.config.coir_datasets.hf_load_dataset")
    def test_load_coir_includes_relevance(self, mock_load: MagicMock) -> None:
        """CoIR queries should include relevant_doc_ids."""
        from embedding_tests.config.coir_datasets import load_coir_dataset

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = lambda self: iter([
            {"_id": "func1", "text": "code"},
        ])
        mock_corpus.__len__ = lambda self: 1

        mock_queries = MagicMock()
        mock_queries.__iter__ = lambda self: iter([
            {"_id": "q1", "text": "Query 1"},
        ])
        mock_queries.__len__ = lambda self: 1

        mock_qrels = MagicMock()
        mock_qrels.__iter__ = lambda self: iter([
            {"query_id": "q1", "positive_passages": [{"docid": "func1"}], "negative_passages": []},
        ])

        def side_effect(name, subset=None, **kwargs):
            if subset and "corpus" in subset:
                return {"train": mock_corpus}
            elif subset and "queries" in subset:
                return {"train": mock_queries}
            elif subset and "qrels" in subset:
                return {"train": mock_qrels}
            return {}

        mock_load.side_effect = side_effect

        corpus, queries = load_coir_dataset("codesearchnet-python")

        assert queries[0]["relevant_doc_ids"] == ["func1"]

    def test_load_coir_invalid_dataset_raises(self) -> None:
        """Should raise for unknown CoIR dataset."""
        from embedding_tests.config.coir_datasets import load_coir_dataset

        with pytest.raises(ValueError, match="Unknown CoIR dataset"):
            load_coir_dataset("nonexistent_dataset")

    @patch("embedding_tests.config.coir_datasets.hf_load_dataset")
    def test_load_coir_with_limit(self, mock_load: MagicMock) -> None:
        """Should support limiting corpus and query count."""
        from embedding_tests.config.coir_datasets import load_coir_dataset

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = lambda self: iter([
            {"_id": f"func{i}", "text": f"code {i}"}
            for i in range(100)
        ])
        mock_corpus.__len__ = lambda self: 100

        mock_queries = MagicMock()
        mock_queries.__iter__ = lambda self: iter([
            {"_id": f"q{i}", "text": f"Query {i}"}
            for i in range(50)
        ])
        mock_queries.__len__ = lambda self: 50

        mock_qrels = MagicMock()
        mock_qrels.__iter__ = lambda self: iter([])

        def side_effect(name, subset=None, **kwargs):
            if subset and "corpus" in subset:
                return {"train": mock_corpus}
            elif subset and "queries" in subset:
                return {"train": mock_queries}
            elif subset and "qrels" in subset:
                return {"train": mock_qrels}
            return {}

        mock_load.side_effect = side_effect

        corpus, queries = load_coir_dataset(
            "codesearchnet-python", max_corpus=10, max_queries=5
        )

        assert len(corpus) == 10
        assert len(queries) == 5

    def test_list_coir_datasets_function(self) -> None:
        """Should have a function to list datasets with info."""
        from embedding_tests.config.coir_datasets import list_coir_datasets

        datasets = list_coir_datasets()
        assert len(datasets) == 6
        assert all("name" in d and "language" in d for d in datasets)

    def test_is_coir_dataset(self) -> None:
        """Should detect CoIR dataset names."""
        from embedding_tests.config.coir_datasets import is_coir_dataset

        assert is_coir_dataset("codesearchnet-python")
        assert is_coir_dataset("codesearchnet-java")
        assert not is_coir_dataset("python")
        assert not is_coir_dataset("sample")
