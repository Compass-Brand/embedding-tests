"""Dataset loading for experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SAMPLE_DIR = _PROJECT_ROOT / "tests" / "fixtures"


def load_dataset(
    name: str | None = None,
    *,
    data_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load corpus and queries for an experiment.

    Args:
        name: Dataset name. ``"sample"`` or ``None`` loads the built-in
              sample fixtures.  Other names resolve to
              ``{data_dir}/{name}/corpus.json`` and ``queries.json``.
        data_dir: Base directory for custom datasets.

    Returns:
        Tuple of (corpus, queries) where corpus items have ``doc_id``
        and ``text`` keys, and query items have ``query_id``, ``text``,
        and ``relevant_doc_ids`` keys.
    """
    if name is None or name == "sample":
        corpus_path = _SAMPLE_DIR / "sample_corpus.json"
        queries_path = _SAMPLE_DIR / "sample_queries.json"
    else:
        base = data_dir or (_PROJECT_ROOT / "data")
        ds_dir = base / name
        corpus_path = ds_dir / "corpus.json"
        queries_path = ds_dir / "queries.json"

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found for dataset {name!r}: {corpus_path}"
        )
    if not queries_path.exists():
        raise FileNotFoundError(
            f"Queries file not found for dataset {name!r}: {queries_path}"
        )

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    _validate_corpus(corpus)
    _validate_queries(queries)

    return corpus, queries


def _validate_corpus(corpus: list[dict[str, Any]]) -> None:
    for i, doc in enumerate(corpus):
        if "doc_id" not in doc:
            raise ValueError(f"Corpus item {i} missing required 'doc_id' field")
        if "text" not in doc:
            raise ValueError(f"Corpus item {i} missing required 'text' field")


def _validate_queries(queries: list[dict[str, Any]]) -> None:
    for i, q in enumerate(queries):
        if "query_id" not in q:
            raise ValueError(f"Query item {i} missing required 'query_id' field")
        if "text" not in q:
            raise ValueError(f"Query item {i} missing required 'text' field")
