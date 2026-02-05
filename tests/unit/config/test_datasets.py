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
