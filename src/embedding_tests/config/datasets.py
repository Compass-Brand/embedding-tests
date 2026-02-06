"""Dataset loading for experiments.

Unified loader that auto-routes to appropriate backend based on dataset name.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from embedding_tests.config.beir_datasets import (
    BEIR_DATASETS,
    is_beir_dataset,
    list_beir_datasets,
    load_beir_dataset,
)
from embedding_tests.config.coir_datasets import (
    COIR_DATASETS,
    is_coir_dataset,
    list_coir_datasets,
    load_coir_dataset,
)
from embedding_tests.config.mteb_datasets import (
    MTEB_RETRIEVAL_TASKS,
    is_mteb_dataset,
    list_mteb_datasets,
    load_mteb_dataset,
)
from embedding_tests.config.nanobeir_datasets import (
    NANOBEIR_DATASETS,
    is_nanobeir_dataset,
    list_nanobeir_datasets,
    load_nanobeir_dataset,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SAMPLE_DIR = _PROJECT_ROOT / "tests" / "fixtures"


# Dataset categories for filtering
DATASET_CATEGORIES: dict[str, set[str]] = {
    "nano": set(NANOBEIR_DATASETS.keys()),
    "beir": set(BEIR_DATASETS.keys()),
    "code": set(COIR_DATASETS.keys()),
    "technical": set(MTEB_RETRIEVAL_TASKS.keys()),
    "scientific": {"nfcorpus", "scifact", "trec-covid", "nano-nfcorpus", "nano-scifact"},
}


def list_all_datasets(category: str | None = None) -> list[dict[str, Any]]:
    """List all available datasets with metadata.

    Args:
        category: Optional category filter: "nano", "beir", "code", "technical",
                 "scientific". If None, returns all datasets.

    Returns:
        List of dicts with 'name', 'category', and other metadata.
    """
    datasets = []

    # Sample dataset
    datasets.append({
        "name": "sample",
        "category": "sample",
        "description": "Built-in sample fixtures (10 docs, 5 queries)",
    })

    # NanoBEIR datasets
    for ds in list_nanobeir_datasets():
        datasets.append({
            "name": ds["name"],
            "category": "nano",
            "hf_name": ds["hf_name"],
            "description": f"Small benchmark: {ds['name'].replace('nano-', '')}",
        })

    # BEIR datasets
    for ds in list_beir_datasets():
        datasets.append({
            "name": ds["name"],
            "category": "beir",
            "hf_name": ds["hf_name"],
            "description": ds["description"],
        })

    # CoIR datasets
    for ds in list_coir_datasets():
        datasets.append({
            "name": ds["name"],
            "category": "code",
            "hf_name": ds["hf_name"],
            "language": ds["language"],
            "description": f"Code retrieval: {ds['language']}",
        })

    # MTEB datasets
    for ds in list_mteb_datasets():
        datasets.append({
            "name": ds["name"],
            "category": "technical",
            "task_name": ds["task_name"],
            "description": f"MTEB: {ds['task_name']}",
        })

    # Filter by category if specified
    if category is not None:
        if category not in DATASET_CATEGORIES and category != "sample":
            valid = sorted(list(DATASET_CATEGORIES.keys()) + ["sample"])
            raise ValueError(f"Unknown category: {category!r}. Valid: {valid}")

        if category == "sample":
            datasets = [d for d in datasets if d["name"] == "sample"]
        else:
            category_names = DATASET_CATEGORIES.get(category, set())
            datasets = [d for d in datasets if d["name"] in category_names]

    return datasets


def load_dataset(
    name: str | None = None,
    *,
    data_dir: Path | None = None,
    max_corpus: int | None = None,
    max_queries: int | None = None,
    split: str = "test",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load corpus and queries for an experiment.

    Auto-routes to the appropriate loader based on dataset name.

    Args:
        name: Dataset name. Supported formats:
            - ``"sample"`` or ``None``: Built-in sample fixtures
            - ``"nano-*"``: NanoBEIR datasets (e.g., "nano-nfcorpus")
            - BEIR names: Standard BEIR datasets (e.g., "nfcorpus", "scifact")
            - ``"codesearchnet-*"``: CodeSearchNet datasets
            - ``"cqadupstack-*"``: CQADupStack technical Q&A datasets
            - Other: Local dataset from ``{data_dir}/{name}/``
        data_dir: Base directory for custom local datasets.
        max_corpus: Maximum number of corpus documents to load.
        max_queries: Maximum number of queries to load.
        split: Dataset split to use for HuggingFace datasets.

    Returns:
        Tuple of (corpus, queries) where corpus items have ``doc_id``
        and ``text`` keys, and query items have ``query_id``, ``text``,
        and ``relevant_doc_ids`` keys.
    """
    # Sample dataset (built-in fixtures)
    if name is None or name == "sample":
        return _load_sample_dataset()

    # NanoBEIR datasets
    if is_nanobeir_dataset(name):
        return load_nanobeir_dataset(
            name, max_corpus=max_corpus, max_queries=max_queries
        )

    # BEIR datasets
    if is_beir_dataset(name):
        return load_beir_dataset(
            name, split=split, max_corpus=max_corpus, max_queries=max_queries
        )

    # CoIR (CodeSearchNet) datasets
    if is_coir_dataset(name):
        return load_coir_dataset(
            name, split=split, max_corpus=max_corpus, max_queries=max_queries
        )

    # MTEB retrieval datasets
    if is_mteb_dataset(name):
        return load_mteb_dataset(
            name, split=split, max_corpus=max_corpus, max_queries=max_queries
        )

    # Local dataset (fallback)
    logger.debug("Loading local dataset: %s", name)
    return _load_local_dataset(name, data_dir=data_dir)


def _load_sample_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the built-in sample fixtures."""
    corpus_path = _SAMPLE_DIR / "sample_corpus.json"
    queries_path = _SAMPLE_DIR / "sample_queries.json"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Sample corpus not found: {corpus_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Sample queries not found: {queries_path}")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    _validate_corpus(corpus)
    _validate_queries(queries)

    return corpus, queries


def _load_local_dataset(
    name: str,
    *,
    data_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a local dataset from corpus.json and queries.json files."""
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
