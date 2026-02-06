"""Tests for NanoBEIR dataset loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestNanoBEIRDatasetLoader:
    """Tests for NanoBEIR dataset loading from sentence-transformers."""

    def test_list_available_nanobeir_datasets(self) -> None:
        """Should list available NanoBEIR datasets."""
        from embedding_tests.config.nanobeir_datasets import NANOBEIR_DATASETS

        expected = {
            "nano-nfcorpus",
            "nano-scifact",
            "nano-fiqa",
            "nano-arguana",
            "nano-scidocs",
            "nano-quora",
        }
        assert set(NANOBEIR_DATASETS.keys()) == expected

    def test_nanobeir_dataset_has_hf_name(self) -> None:
        """Each NanoBEIR dataset should have HuggingFace name."""
        from embedding_tests.config.nanobeir_datasets import NANOBEIR_DATASETS

        for name, hf_name in NANOBEIR_DATASETS.items():
            assert hf_name.startswith("sentence-transformers/Nano"), \
                f"{name} should have sentence-transformers HF name"

    @patch("embedding_tests.config.nanobeir_datasets.hf_load_dataset")
    def test_load_nanobeir_corpus_converts_format(self, mock_load: MagicMock) -> None:
        """NanoBEIR corpus should be converted to our format."""
        from embedding_tests.config.nanobeir_datasets import load_nanobeir_dataset

        # NanoBEIR format: corpus has _id, title, text; queries has _id, text
        mock_corpus = MagicMock()
        mock_corpus.__iter__ = lambda self: iter([
            {"_id": "doc1", "title": "Title 1", "text": "Text 1"},
            {"_id": "doc2", "title": "Title 2", "text": "Text 2"},
        ])
        mock_corpus.__len__ = lambda self: 2

        mock_queries = MagicMock()
        mock_queries.__iter__ = lambda self: iter([
            {"_id": "q1", "text": "Query 1"},
        ])
        mock_queries.__len__ = lambda self: 1

        mock_qrels = MagicMock()
        mock_qrels.__iter__ = lambda self: iter([
            {"query-id": "q1", "corpus-id": "doc1", "score": 1},
        ])

        mock_load.return_value = {
            "corpus": mock_corpus,
            "queries": mock_queries,
            "qrels": mock_qrels,
        }

        corpus, queries = load_nanobeir_dataset("nano-nfcorpus")

        assert len(corpus) == 2
        assert corpus[0]["doc_id"] == "doc1"
        assert "Title 1" in corpus[0]["text"]
        assert "Text 1" in corpus[0]["text"]

    @patch("embedding_tests.config.nanobeir_datasets.hf_load_dataset")
    def test_load_nanobeir_queries_converts_format(self, mock_load: MagicMock) -> None:
        """NanoBEIR queries should be converted to our format."""
        from embedding_tests.config.nanobeir_datasets import load_nanobeir_dataset

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

        mock_load.return_value = {
            "corpus": mock_corpus,
            "queries": mock_queries,
            "qrels": mock_qrels,
        }

        corpus, queries = load_nanobeir_dataset("nano-nfcorpus")

        assert len(queries) == 2
        assert queries[0]["query_id"] == "q1"
        assert queries[0]["text"] == "Query 1"

    @patch("embedding_tests.config.nanobeir_datasets.hf_load_dataset")
    def test_load_nanobeir_includes_relevance(self, mock_load: MagicMock) -> None:
        """NanoBEIR queries should include relevant_doc_ids."""
        from embedding_tests.config.nanobeir_datasets import load_nanobeir_dataset

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = lambda self: iter([
            {"_id": "doc1", "title": "", "text": "Text"},
        ])
        mock_corpus.__len__ = lambda self: 1

        mock_queries = MagicMock()
        mock_queries.__iter__ = lambda self: iter([
            {"_id": "q1", "text": "Query 1"},
        ])
        mock_queries.__len__ = lambda self: 1

        mock_qrels = MagicMock()
        mock_qrels.__iter__ = lambda self: iter([
            {"query-id": "q1", "corpus-id": "doc1", "score": 1},
        ])

        mock_load.return_value = {
            "corpus": mock_corpus,
            "queries": mock_queries,
            "qrels": mock_qrels,
        }

        corpus, queries = load_nanobeir_dataset("nano-nfcorpus")

        assert queries[0]["relevant_doc_ids"] == ["doc1"]

    def test_load_nanobeir_invalid_dataset_raises(self) -> None:
        """Should raise for unknown NanoBEIR dataset."""
        from embedding_tests.config.nanobeir_datasets import load_nanobeir_dataset

        with pytest.raises(ValueError, match="Unknown NanoBEIR dataset"):
            load_nanobeir_dataset("nonexistent_dataset")

    @patch("embedding_tests.config.nanobeir_datasets.hf_load_dataset")
    def test_load_nanobeir_with_limit(self, mock_load: MagicMock) -> None:
        """Should support limiting corpus and query count."""
        from embedding_tests.config.nanobeir_datasets import load_nanobeir_dataset

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = lambda self: iter([
            {"_id": f"doc{i}", "title": "", "text": f"Text {i}"}
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

        mock_load.return_value = {
            "corpus": mock_corpus,
            "queries": mock_queries,
            "qrels": mock_qrels,
        }

        corpus, queries = load_nanobeir_dataset(
            "nano-nfcorpus", max_corpus=10, max_queries=5
        )

        assert len(corpus) == 10
        assert len(queries) == 5

    @patch("embedding_tests.config.nanobeir_datasets.hf_load_dataset")
    def test_load_nanobeir_handles_empty_title(self, mock_load: MagicMock) -> None:
        """Empty title should not produce spurious newlines in text."""
        from embedding_tests.config.nanobeir_datasets import load_nanobeir_dataset

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = lambda self: iter([
            {"_id": "doc1", "title": "", "text": "Just text"},
            {"_id": "doc2", "title": "  ", "text": "More text"},
        ])
        mock_corpus.__len__ = lambda self: 2

        mock_queries = MagicMock()
        mock_queries.__iter__ = lambda self: iter([{"_id": "q1", "text": "Query"}])
        mock_queries.__len__ = lambda self: 1

        mock_qrels = MagicMock()
        mock_qrels.__iter__ = lambda self: iter([])

        mock_load.return_value = {
            "corpus": mock_corpus,
            "queries": mock_queries,
            "qrels": mock_qrels,
        }

        corpus, queries = load_nanobeir_dataset("nano-nfcorpus")

        assert corpus[0]["text"] == "Just text"
        assert corpus[1]["text"] == "More text"

    def test_list_nanobeir_datasets_function(self) -> None:
        """Should have a function to list datasets with info."""
        from embedding_tests.config.nanobeir_datasets import list_nanobeir_datasets

        datasets = list_nanobeir_datasets()
        assert len(datasets) == 6
        assert all("name" in d and "hf_name" in d for d in datasets)

    def test_is_nanobeir_dataset(self) -> None:
        """Should detect NanoBEIR dataset names."""
        from embedding_tests.config.nanobeir_datasets import is_nanobeir_dataset

        assert is_nanobeir_dataset("nano-nfcorpus")
        assert is_nanobeir_dataset("nano-scifact")
        assert not is_nanobeir_dataset("nfcorpus")
        assert not is_nanobeir_dataset("sample")
