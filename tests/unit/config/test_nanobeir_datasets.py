"""Tests for NanoBEIR dataset loader."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

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

    def test_nanobeir_dataset_has_split_name(self) -> None:
        """Each NanoBEIR dataset should map to a split name."""
        from embedding_tests.config.nanobeir_datasets import NANOBEIR_DATASETS

        for name, split_name in NANOBEIR_DATASETS.items():
            # Split names are like "NanoNFCorpus", "NanoSciFact", etc.
            assert split_name.startswith("Nano"), \
                f"{name} should have Nano* split name, got {split_name}"

    def test_nanobeir_hf_dataset_constant(self) -> None:
        """Should use the unified NanoBEIR-en dataset."""
        from embedding_tests.config.nanobeir_datasets import NANOBEIR_HF_DATASET

        assert NANOBEIR_HF_DATASET == "sentence-transformers/NanoBEIR-en"

    def _create_mock_dataset(self, data: list[dict]) -> MagicMock:
        """Create a mock HuggingFace Dataset."""
        mock = MagicMock()
        mock.__iter__ = lambda self: iter(data)
        mock.__len__ = lambda self: len(data)
        return mock

    @patch("embedding_tests.config.nanobeir_datasets.hf_load_dataset")
    def test_load_nanobeir_corpus_converts_format(self, mock_load: MagicMock) -> None:
        """NanoBEIR corpus should be converted to our format."""
        from embedding_tests.config.nanobeir_datasets import load_nanobeir_dataset

        # NanoBEIR structure: corpus has _id, text; queries has _id, text
        mock_corpus = self._create_mock_dataset([
            {"_id": "doc1", "text": "Text 1"},
            {"_id": "doc2", "text": "Text 2"},
        ])
        mock_queries = self._create_mock_dataset([
            {"_id": "q1", "text": "Query 1"},
        ])
        mock_qrels = self._create_mock_dataset([
            {"query-id": "q1", "corpus-id": "doc1", "score": 1},
        ])

        # Each config call returns a DatasetDict with splits
        mock_load.side_effect = [
            {"NanoNFCorpus": mock_corpus},  # corpus config
            {"NanoNFCorpus": mock_queries},  # queries config
            {"NanoNFCorpus": mock_qrels},   # qrels config
        ]

        corpus, queries = load_nanobeir_dataset("nano-nfcorpus")

        # Verify hf_load_dataset was called for each config
        assert mock_load.call_count == 3
        mock_load.assert_any_call("sentence-transformers/NanoBEIR-en", "corpus")
        mock_load.assert_any_call("sentence-transformers/NanoBEIR-en", "queries")
        mock_load.assert_any_call("sentence-transformers/NanoBEIR-en", "qrels")

        assert len(corpus) == 2
        assert corpus[0]["doc_id"] == "doc1"
        assert corpus[0]["text"] == "Text 1"

    @patch("embedding_tests.config.nanobeir_datasets.hf_load_dataset")
    def test_load_nanobeir_queries_converts_format(self, mock_load: MagicMock) -> None:
        """NanoBEIR queries should be converted to our format."""
        from embedding_tests.config.nanobeir_datasets import load_nanobeir_dataset

        mock_corpus = self._create_mock_dataset([])
        mock_queries = self._create_mock_dataset([
            {"_id": "q1", "text": "Query 1"},
            {"_id": "q2", "text": "Query 2"},
        ])
        mock_qrels = self._create_mock_dataset([])

        mock_load.side_effect = [
            {"NanoNFCorpus": mock_corpus},
            {"NanoNFCorpus": mock_queries},
            {"NanoNFCorpus": mock_qrels},
        ]

        corpus, queries = load_nanobeir_dataset("nano-nfcorpus")

        assert len(queries) == 2
        assert queries[0]["query_id"] == "q1"
        assert queries[0]["text"] == "Query 1"

    @patch("embedding_tests.config.nanobeir_datasets.hf_load_dataset")
    def test_load_nanobeir_includes_relevance(self, mock_load: MagicMock) -> None:
        """NanoBEIR queries should include relevant_doc_ids."""
        from embedding_tests.config.nanobeir_datasets import load_nanobeir_dataset

        mock_corpus = self._create_mock_dataset([
            {"_id": "doc1", "text": "Text"},
        ])
        mock_queries = self._create_mock_dataset([
            {"_id": "q1", "text": "Query 1"},
        ])
        mock_qrels = self._create_mock_dataset([
            {"query-id": "q1", "corpus-id": "doc1", "score": 1},
        ])

        mock_load.side_effect = [
            {"NanoNFCorpus": mock_corpus},
            {"NanoNFCorpus": mock_queries},
            {"NanoNFCorpus": mock_qrels},
        ]

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

        mock_corpus = self._create_mock_dataset([
            {"_id": f"doc{i}", "text": f"Text {i}"}
            for i in range(100)
        ])
        mock_queries = self._create_mock_dataset([
            {"_id": f"q{i}", "text": f"Query {i}"}
            for i in range(50)
        ])
        mock_qrels = self._create_mock_dataset([])

        mock_load.side_effect = [
            {"NanoNFCorpus": mock_corpus},
            {"NanoNFCorpus": mock_queries},
            {"NanoNFCorpus": mock_qrels},
        ]

        corpus, queries = load_nanobeir_dataset(
            "nano-nfcorpus", max_corpus=10, max_queries=5
        )

        assert len(corpus) == 10
        assert len(queries) == 5

    @patch("embedding_tests.config.nanobeir_datasets.hf_load_dataset")
    def test_load_nanobeir_handles_title_in_corpus(self, mock_load: MagicMock) -> None:
        """Should handle corpus with optional title field."""
        from embedding_tests.config.nanobeir_datasets import load_nanobeir_dataset

        mock_corpus = self._create_mock_dataset([
            {"_id": "doc1", "title": "Title 1", "text": "Text 1"},
            {"_id": "doc2", "text": "Just text"},  # No title
        ])
        mock_queries = self._create_mock_dataset([{"_id": "q1", "text": "Query"}])
        mock_qrels = self._create_mock_dataset([])

        mock_load.side_effect = [
            {"NanoNFCorpus": mock_corpus},
            {"NanoNFCorpus": mock_queries},
            {"NanoNFCorpus": mock_qrels},
        ]

        corpus, queries = load_nanobeir_dataset("nano-nfcorpus")

        assert "Title 1" in corpus[0]["text"]
        assert corpus[1]["text"] == "Just text"

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

    @patch("embedding_tests.config.nanobeir_datasets.hf_load_dataset")
    def test_load_nanobeir_handles_id_format(self, mock_load: MagicMock) -> None:
        """Should handle datasets with id instead of _id."""
        from embedding_tests.config.nanobeir_datasets import load_nanobeir_dataset

        mock_corpus = self._create_mock_dataset([
            {"id": "doc1", "text": "Text"},
        ])
        mock_queries = self._create_mock_dataset([
            {"id": "q1", "text": "Query 1"},
        ])
        mock_qrels = self._create_mock_dataset([])

        mock_load.side_effect = [
            {"NanoNFCorpus": mock_corpus},
            {"NanoNFCorpus": mock_queries},
            {"NanoNFCorpus": mock_qrels},
        ]

        corpus, queries = load_nanobeir_dataset("nano-nfcorpus")

        assert corpus[0]["doc_id"] == "doc1"
        assert queries[0]["query_id"] == "q1"
