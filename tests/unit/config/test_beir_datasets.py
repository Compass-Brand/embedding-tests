"""Tests for BEIR dataset loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestBEIRDatasetLoader:
    """Tests for BEIR dataset loading from HuggingFace."""

    def test_list_available_beir_datasets(self) -> None:
        """Should list available BEIR datasets."""
        from embedding_tests.config.beir_datasets import BEIR_DATASETS

        assert "nfcorpus" in BEIR_DATASETS
        assert "scifact" in BEIR_DATASETS
        assert "fiqa" in BEIR_DATASETS
        assert "hotpotqa" in BEIR_DATASETS

    def test_beir_dataset_info_has_required_fields(self) -> None:
        """Each BEIR dataset should have metadata."""
        from embedding_tests.config.beir_datasets import BEIR_DATASETS

        for name, info in BEIR_DATASETS.items():
            assert "hf_name" in info, f"{name} missing hf_name"
            assert "description" in info, f"{name} missing description"

    @patch("datasets.load_dataset")
    def test_load_beir_corpus_converts_format(self, mock_load: MagicMock) -> None:
        """BEIR corpus should be converted to our format."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        # Mock HuggingFace dataset response
        mock_corpus = [
            {"_id": "doc1", "title": "Title 1", "text": "Text 1"},
            {"_id": "doc2", "title": "Title 2", "text": "Text 2"},
        ]
        mock_queries = [
            {"_id": "q1", "text": "Query 1"},
        ]
        mock_qrels = [
            {"query-id": "q1", "corpus-id": "doc1", "score": 1},
        ]

        def side_effect(name, *args, **kwargs):
            if "qrels" in args or kwargs.get("name") == "qrels":
                return {"test": mock_qrels}
            return {"corpus": mock_corpus, "queries": mock_queries}

        mock_load.side_effect = side_effect

        corpus, queries = load_beir_dataset("nfcorpus")

        assert len(corpus) == 2
        assert corpus[0]["doc_id"] == "doc1"
        assert "Title 1" in corpus[0]["text"]
        assert "Text 1" in corpus[0]["text"]

    @patch("datasets.load_dataset")
    def test_load_beir_queries_converts_format(self, mock_load: MagicMock) -> None:
        """BEIR queries should be converted to our format."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        mock_corpus: list[dict] = []
        mock_queries = [
            {"_id": "q1", "text": "Query 1"},
            {"_id": "q2", "text": "Query 2"},
        ]
        mock_qrels: list[dict] = []

        def side_effect(name, *args, **kwargs):
            if "qrels" in args:
                return {"test": mock_qrels}
            return {"corpus": mock_corpus, "queries": mock_queries}

        mock_load.side_effect = side_effect

        corpus, queries = load_beir_dataset("nfcorpus")

        assert len(queries) == 2
        assert queries[0]["query_id"] == "q1"
        assert queries[0]["text"] == "Query 1"

    @patch("datasets.load_dataset")
    def test_load_beir_includes_relevance_judgments(self, mock_load: MagicMock) -> None:
        """BEIR queries should include relevant_doc_ids from qrels."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        mock_corpus = [
            {"_id": "doc1", "title": "", "text": "Text"},
        ]
        mock_queries = [
            {"_id": "q1", "text": "Query 1"},
        ]
        mock_qrels = [
            {"query-id": "q1", "corpus-id": "doc1", "score": 1},
        ]

        def side_effect(name, *args, **kwargs):
            if "qrels" in args:
                return {"test": mock_qrels}
            return {"corpus": mock_corpus, "queries": mock_queries}

        mock_load.side_effect = side_effect

        corpus, queries = load_beir_dataset("nfcorpus")

        assert queries[0]["relevant_doc_ids"] == ["doc1"]

    def test_load_beir_invalid_dataset_raises(self) -> None:
        """Should raise for unknown BEIR dataset."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        with pytest.raises(ValueError, match="Unknown BEIR dataset"):
            load_beir_dataset("nonexistent_dataset")

    @patch("datasets.load_dataset")
    def test_load_beir_with_limit(self, mock_load: MagicMock) -> None:
        """Should support limiting corpus and query count."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        mock_corpus = [
            {"_id": f"doc{i}", "title": "", "text": f"Text {i}"}
            for i in range(100)
        ]
        mock_queries = [
            {"_id": f"q{i}", "text": f"Query {i}"}
            for i in range(50)
        ]
        mock_qrels: list[dict] = []

        def side_effect(name, *args, **kwargs):
            if "qrels" in args:
                return {"test": mock_qrels}
            return {"corpus": mock_corpus, "queries": mock_queries}

        mock_load.side_effect = side_effect

        corpus, queries = load_beir_dataset("nfcorpus", max_corpus=10, max_queries=5)

        assert len(corpus) == 10
        assert len(queries) == 5
