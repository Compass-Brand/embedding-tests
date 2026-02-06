"""Tests for dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a custom dataset."""
    ds_dir = tmp_path / "data" / "custom"
    ds_dir.mkdir(parents=True)

    corpus = [
        {"doc_id": "d1", "text": "First document."},
        {"doc_id": "d2", "text": "Second document."},
    ]
    queries = [
        {"query_id": "q1", "text": "Find first.", "relevant_doc_ids": ["d1"]},
    ]
    (ds_dir / "corpus.json").write_text(json.dumps(corpus))
    (ds_dir / "queries.json").write_text(json.dumps(queries))
    return tmp_path / "data"


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_sample_dataset(self) -> None:
        from embedding_tests.config.datasets import load_dataset

        corpus, queries = load_dataset("sample")
        assert len(corpus) == 10
        assert len(queries) == 5
        assert all("doc_id" in d and "text" in d for d in corpus)
        assert all("query_id" in q and "text" in q for q in queries)

    def test_load_custom_dataset(self, data_dir: Path) -> None:
        from embedding_tests.config.datasets import load_dataset

        corpus, queries = load_dataset("custom", data_dir=data_dir)
        assert len(corpus) == 2
        assert len(queries) == 1
        assert corpus[0]["doc_id"] == "d1"
        assert queries[0]["query_id"] == "q1"

    def test_load_dataset_unknown_name_raises(self) -> None:
        from embedding_tests.config.datasets import load_dataset

        with pytest.raises(FileNotFoundError, match="not_a_dataset"):
            load_dataset("not_a_dataset")

    def test_load_dataset_validates_corpus_structure(self, tmp_path: Path) -> None:
        from embedding_tests.config.datasets import load_dataset

        ds_dir = tmp_path / "bad"
        ds_dir.mkdir(parents=True)
        (ds_dir / "corpus.json").write_text(json.dumps([{"no_doc_id": "x"}]))
        (ds_dir / "queries.json").write_text(json.dumps([]))

        with pytest.raises(ValueError, match="doc_id"):
            load_dataset("bad", data_dir=tmp_path)

    def test_load_dataset_validates_queries_structure(self, tmp_path: Path) -> None:
        from embedding_tests.config.datasets import load_dataset

        ds_dir = tmp_path / "bad2"
        ds_dir.mkdir(parents=True)
        (ds_dir / "corpus.json").write_text(
            json.dumps([{"doc_id": "d1", "text": "ok"}])
        )
        (ds_dir / "queries.json").write_text(
            json.dumps([{"missing_fields": True}])
        )

        with pytest.raises(ValueError, match="query_id"):
            load_dataset("bad2", data_dir=tmp_path)

    def test_load_dataset_default_is_sample(self) -> None:
        from embedding_tests.config.datasets import load_dataset

        corpus, queries = load_dataset()
        assert len(corpus) == 10  # Same as sample


class TestListAllDatasets:
    """Tests for list_all_datasets function."""

    def test_list_all_datasets_includes_all_types(self) -> None:
        from embedding_tests.config.datasets import list_all_datasets

        datasets = list_all_datasets()
        # Should include sample, nano, beir, code, technical
        names = {d["name"] for d in datasets}
        assert "sample" in names
        assert "nano-nfcorpus" in names
        assert "nfcorpus" in names
        assert "codesearchnet-python" in names
        assert "cqadupstack-programmers" in names

    def test_list_all_datasets_with_category_filter(self) -> None:
        from embedding_tests.config.datasets import list_all_datasets

        nano_datasets = list_all_datasets(category="nano")
        assert all(d["category"] == "nano" for d in nano_datasets)
        assert all(d["name"].startswith("nano-") for d in nano_datasets)

    def test_list_all_datasets_invalid_category_raises(self) -> None:
        from embedding_tests.config.datasets import list_all_datasets

        with pytest.raises(ValueError, match="Unknown category"):
            list_all_datasets(category="invalid")

    def test_list_all_datasets_has_description(self) -> None:
        from embedding_tests.config.datasets import list_all_datasets

        datasets = list_all_datasets()
        assert all("description" in d for d in datasets)


class TestUnifiedDatasetRouting:
    """Tests for unified dataset routing."""

    def test_routes_nanobeir_dataset(self) -> None:
        from unittest.mock import patch

        with patch("embedding_tests.config.datasets.load_nanobeir_dataset") as mock:
            mock.return_value = ([], [])
            from embedding_tests.config.datasets import load_dataset
            load_dataset("nano-nfcorpus")
            mock.assert_called_once()

    def test_routes_beir_dataset(self) -> None:
        from unittest.mock import patch

        with patch("embedding_tests.config.datasets.load_beir_dataset") as mock:
            mock.return_value = ([], [])
            from embedding_tests.config.datasets import load_dataset
            load_dataset("nfcorpus")
            mock.assert_called_once()

    def test_routes_coir_dataset(self) -> None:
        from unittest.mock import patch

        with patch("embedding_tests.config.datasets.load_coir_dataset") as mock:
            mock.return_value = ([], [])
            from embedding_tests.config.datasets import load_dataset
            load_dataset("codesearchnet-python")
            mock.assert_called_once()

    def test_routes_mteb_dataset(self) -> None:
        from unittest.mock import patch

        with patch("embedding_tests.config.datasets.load_mteb_dataset") as mock:
            mock.return_value = ([], [])
            from embedding_tests.config.datasets import load_dataset
            load_dataset("cqadupstack-programmers")
            mock.assert_called_once()


class TestCacheDirRouting:
    """Tests for cache_dir parameter routing to sub-loaders."""

    def test_routes_cache_dir_to_nanobeir(self) -> None:
        from pathlib import Path
        from unittest.mock import patch

        with patch("embedding_tests.config.datasets.load_nanobeir_dataset") as mock:
            mock.return_value = ([], [])
            from embedding_tests.config.datasets import load_dataset
            cache_dir = Path("/custom/cache")
            load_dataset("nano-nfcorpus", cache_dir=cache_dir)
            mock.assert_called_once()
            assert mock.call_args.kwargs.get("cache_dir") == cache_dir

    def test_routes_cache_dir_to_beir(self) -> None:
        from pathlib import Path
        from unittest.mock import patch

        with patch("embedding_tests.config.datasets.load_beir_dataset") as mock:
            mock.return_value = ([], [])
            from embedding_tests.config.datasets import load_dataset
            cache_dir = Path("/custom/cache")
            load_dataset("nfcorpus", cache_dir=cache_dir)
            mock.assert_called_once()
            assert mock.call_args.kwargs.get("cache_dir") == cache_dir

    def test_routes_cache_dir_to_coir(self) -> None:
        from pathlib import Path
        from unittest.mock import patch

        with patch("embedding_tests.config.datasets.load_coir_dataset") as mock:
            mock.return_value = ([], [])
            from embedding_tests.config.datasets import load_dataset
            cache_dir = Path("/custom/cache")
            load_dataset("codesearchnet-python", cache_dir=cache_dir)
            mock.assert_called_once()
            assert mock.call_args.kwargs.get("cache_dir") == cache_dir

    def test_routes_cache_dir_to_mteb(self) -> None:
        from pathlib import Path
        from unittest.mock import patch

        with patch("embedding_tests.config.datasets.load_mteb_dataset") as mock:
            mock.return_value = ([], [])
            from embedding_tests.config.datasets import load_dataset
            cache_dir = Path("/custom/cache")
            load_dataset("cqadupstack-programmers", cache_dir=cache_dir)
            mock.assert_called_once()
            assert mock.call_args.kwargs.get("cache_dir") == cache_dir
