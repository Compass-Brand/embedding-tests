"""Tests for BEIR dataset loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestBEIRDatasetLoader:
    """Tests for BEIR dataset loading from HuggingFace (via MTEB datasets)."""

    def test_list_available_beir_datasets(self) -> None:
        """Should list available BEIR datasets."""
        from embedding_tests.config.beir_datasets import BEIR_DATASETS

        assert "nfcorpus" in BEIR_DATASETS
        assert "scifact" in BEIR_DATASETS
        assert "fiqa" in BEIR_DATASETS
        assert "hotpotqa" in BEIR_DATASETS
        assert "scidocs" in BEIR_DATASETS

    def test_beir_dataset_info_has_required_fields(self) -> None:
        """Each BEIR dataset should have metadata."""
        from embedding_tests.config.beir_datasets import BEIR_DATASETS

        for name, info in BEIR_DATASETS.items():
            assert "hf_name" in info, f"{name} missing hf_name"
            assert "description" in info, f"{name} missing description"

    def test_beir_dataset_uses_mteb_hf_names(self) -> None:
        """BEIR datasets should use MTEB's HuggingFace names."""
        from embedding_tests.config.beir_datasets import BEIR_DATASETS

        for name, info in BEIR_DATASETS.items():
            assert info["hf_name"].startswith("mteb/"), \
                f"{name} should use mteb/ HuggingFace datasets"

    def _create_mock_dataset(self, data: list[dict]) -> MagicMock:
        """Create a mock HuggingFace Dataset."""
        mock = MagicMock()
        mock.__iter__ = lambda self: iter(data)
        mock.__len__ = lambda self: len(data)
        return mock

    @patch("datasets.load_dataset")
    def test_load_beir_corpus_converts_format(self, mock_load: MagicMock) -> None:
        """BEIR corpus should be converted to our format."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        # Mock corpus, queries, and qrels as separate configs
        mock_corpus = self._create_mock_dataset([
            {"_id": "doc1", "title": "Title 1", "text": "Text 1"},
            {"_id": "doc2", "title": "Title 2", "text": "Text 2"},
        ])
        mock_queries = self._create_mock_dataset([
            {"_id": "q1", "text": "Query 1"},
        ])
        mock_qrels = self._create_mock_dataset([
            {"query-id": "q1", "corpus-id": "doc1", "score": 1},
        ])

        # Each config call returns a DatasetDict
        mock_load.side_effect = [
            {"train": mock_corpus},   # corpus config
            {"queries": mock_queries},  # queries config
            {"test": mock_qrels},       # default (qrels) config
        ]

        corpus, queries = load_beir_dataset("nfcorpus")

        # Verify load_dataset was called for each config
        assert mock_load.call_count == 3
        mock_load.assert_any_call("mteb/nfcorpus", "corpus", cache_dir=None)
        mock_load.assert_any_call("mteb/nfcorpus", "queries", cache_dir=None)
        mock_load.assert_any_call("mteb/nfcorpus", "default", cache_dir=None)

        assert len(corpus) == 2
        assert corpus[0]["doc_id"] == "doc1"
        assert "Title 1" in corpus[0]["text"]
        assert "Text 1" in corpus[0]["text"]

    @patch("datasets.load_dataset")
    def test_load_beir_queries_converts_format(self, mock_load: MagicMock) -> None:
        """BEIR queries should be converted to our format."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        mock_corpus = self._create_mock_dataset([])
        mock_queries = self._create_mock_dataset([
            {"_id": "q1", "text": "Query 1"},
            {"_id": "q2", "text": "Query 2"},
        ])
        mock_qrels = self._create_mock_dataset([])

        mock_load.side_effect = [
            {"train": mock_corpus},
            {"queries": mock_queries},
            {"test": mock_qrels},
        ]

        corpus, queries = load_beir_dataset("nfcorpus")

        assert len(queries) == 2
        assert queries[0]["query_id"] == "q1"
        assert queries[0]["text"] == "Query 1"

    @patch("datasets.load_dataset")
    def test_load_beir_includes_relevance_judgments(self, mock_load: MagicMock) -> None:
        """BEIR queries should include relevant_doc_ids from qrels."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        mock_corpus = self._create_mock_dataset([
            {"_id": "doc1", "title": "", "text": "Text"},
        ])
        mock_queries = self._create_mock_dataset([
            {"_id": "q1", "text": "Query 1"},
        ])
        mock_qrels = self._create_mock_dataset([
            {"query-id": "q1", "corpus-id": "doc1", "score": 1},
        ])

        mock_load.side_effect = [
            {"train": mock_corpus},
            {"queries": mock_queries},
            {"test": mock_qrels},
        ]

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

        mock_corpus = self._create_mock_dataset([
            {"_id": f"doc{i}", "title": "", "text": f"Text {i}"}
            for i in range(100)
        ])
        mock_queries = self._create_mock_dataset([
            {"_id": f"q{i}", "text": f"Query {i}"}
            for i in range(50)
        ])
        mock_qrels = self._create_mock_dataset([])

        mock_load.side_effect = [
            {"train": mock_corpus},
            {"queries": mock_queries},
            {"test": mock_qrels},
        ]

        corpus, queries = load_beir_dataset("nfcorpus", max_corpus=10, max_queries=5)

        assert len(corpus) == 10
        assert len(queries) == 5

    @patch("datasets.load_dataset")
    def test_load_beir_filters_non_positive_relevance(self, mock_load: MagicMock) -> None:
        """Should only include docs with positive relevance scores."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        mock_corpus = self._create_mock_dataset([
            {"_id": "doc1", "title": "", "text": "Text 1"},
            {"_id": "doc2", "title": "", "text": "Text 2"},
        ])
        mock_queries = self._create_mock_dataset([
            {"_id": "q1", "text": "Query"},
        ])
        mock_qrels = self._create_mock_dataset([
            {"query-id": "q1", "corpus-id": "doc1", "score": 1},
            {"query-id": "q1", "corpus-id": "doc2", "score": 0},  # Should be filtered
        ])

        mock_load.side_effect = [
            {"train": mock_corpus},
            {"queries": mock_queries},
            {"test": mock_qrels},
        ]

        corpus, queries = load_beir_dataset("nfcorpus")

        # Only doc1 should be in relevant_doc_ids (score > 0)
        assert queries[0]["relevant_doc_ids"] == ["doc1"]

    @patch("datasets.load_dataset")
    def test_load_beir_handles_empty_title(self, mock_load: MagicMock) -> None:
        """Empty title should not produce spurious newlines in text."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        mock_corpus = self._create_mock_dataset([
            {"_id": "doc1", "title": "", "text": "Just text"},
            {"_id": "doc2", "title": "  ", "text": "More text"},
        ])
        mock_queries = self._create_mock_dataset([
            {"_id": "q1", "text": "Query"},
        ])
        mock_qrels = self._create_mock_dataset([])

        mock_load.side_effect = [
            {"train": mock_corpus},
            {"queries": mock_queries},
            {"test": mock_qrels},
        ]

        corpus, queries = load_beir_dataset("nfcorpus")

        # Text should not start with newlines when title is empty
        assert corpus[0]["text"] == "Just text"
        assert corpus[1]["text"] == "More text"

    @patch("datasets.load_dataset")
    def test_load_beir_raises_on_missing_split(self, mock_load: MagicMock) -> None:
        """Should raise when requested split is not available."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        mock_corpus = self._create_mock_dataset([])
        mock_queries = self._create_mock_dataset([])

        mock_load.side_effect = [
            {"train": mock_corpus},
            {"queries": mock_queries},
            {"train": self._create_mock_dataset([])},  # No test split
        ]

        with pytest.raises(ValueError, match="Split 'test' not found"):
            load_beir_dataset("nfcorpus")

    @patch("datasets.load_dataset")
    def test_load_beir_handles_id_format(self, mock_load: MagicMock) -> None:
        """Should handle datasets with id instead of _id."""
        from embedding_tests.config.beir_datasets import load_beir_dataset

        mock_corpus = self._create_mock_dataset([
            {"id": "doc1", "title": "Title", "text": "Text"},
        ])
        mock_queries = self._create_mock_dataset([
            {"id": "q1", "text": "Query 1"},
        ])
        mock_qrels = self._create_mock_dataset([])

        mock_load.side_effect = [
            {"train": mock_corpus},
            {"queries": mock_queries},
            {"test": mock_qrels},
        ]

        corpus, queries = load_beir_dataset("nfcorpus")

        assert corpus[0]["doc_id"] == "doc1"
        assert queries[0]["query_id"] == "q1"

    @patch("datasets.load_dataset")
    def test_load_beir_passes_cache_dir(self, mock_load: MagicMock) -> None:
        """Should pass cache_dir to load_dataset."""
        from pathlib import Path

        from embedding_tests.config.beir_datasets import load_beir_dataset

        mock_corpus = self._create_mock_dataset([{"_id": "doc1", "title": "", "text": "Text"}])
        mock_queries = self._create_mock_dataset([{"_id": "q1", "text": "Query"}])
        mock_qrels = self._create_mock_dataset([])

        mock_load.side_effect = [
            {"train": mock_corpus},
            {"queries": mock_queries},
            {"test": mock_qrels},
        ]

        cache_dir = Path("/custom/cache")
        load_beir_dataset("nfcorpus", cache_dir=cache_dir)

        # All three calls should include cache_dir
        for call_args in mock_load.call_args_list:
            assert call_args.kwargs.get("cache_dir") == str(cache_dir)
