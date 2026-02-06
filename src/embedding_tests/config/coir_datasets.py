"""CoIR (Code Information Retrieval) dataset loader.

Loads CodeSearchNet datasets for code-to-docstring retrieval.
See: https://github.com/CoIR-team/coir
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Map short names to HuggingFace dataset configs
COIR_DATASETS: dict[str, dict[str, str]] = {
    "codesearchnet-python": {
        "hf_name": "CoIR-Retrieval/codesearchnet",
        "language": "python",
        "subset_prefix": "python",
    },
    "codesearchnet-java": {
        "hf_name": "CoIR-Retrieval/codesearchnet",
        "language": "java",
        "subset_prefix": "java",
    },
    "codesearchnet-javascript": {
        "hf_name": "CoIR-Retrieval/codesearchnet",
        "language": "javascript",
        "subset_prefix": "javascript",
    },
    "codesearchnet-go": {
        "hf_name": "CoIR-Retrieval/codesearchnet",
        "language": "go",
        "subset_prefix": "go",
    },
    "codesearchnet-php": {
        "hf_name": "CoIR-Retrieval/codesearchnet",
        "language": "php",
        "subset_prefix": "php",
    },
    "codesearchnet-ruby": {
        "hf_name": "CoIR-Retrieval/codesearchnet",
        "language": "ruby",
        "subset_prefix": "ruby",
    },
}


def is_coir_dataset(name: str) -> bool:
    """Check if a dataset name is a CoIR dataset."""
    return name in COIR_DATASETS


def list_coir_datasets() -> list[dict[str, str]]:
    """List all available CoIR datasets.

    Returns:
        List of dicts with 'name', 'language', and 'hf_name' keys.
    """
    return [
        {"name": name, "language": info["language"], "hf_name": info["hf_name"]}
        for name, info in COIR_DATASETS.items()
    ]


def hf_load_dataset(name: str, subset: str | None = None, **kwargs: Any) -> Any:
    """Wrapper for HuggingFace datasets.load_dataset for easier mocking."""
    from datasets import load_dataset
    return load_dataset(name, subset, **kwargs)


def load_coir_dataset(
    name: str,
    *,
    split: str = "train",
    max_corpus: int | None = None,
    max_queries: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a CoIR CodeSearchNet dataset from HuggingFace.

    Args:
        name: CoIR dataset name (e.g., "codesearchnet-python").
        split: Dataset split to use (default: "train").
        max_corpus: Maximum number of corpus documents to load.
        max_queries: Maximum number of queries to load.

    Returns:
        Tuple of (corpus, queries) in our standard format.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if name not in COIR_DATASETS:
        raise ValueError(
            f"Unknown CoIR dataset: {name!r}. "
            f"Available: {', '.join(COIR_DATASETS.keys())}"
        )

    info = COIR_DATASETS[name]
    hf_name = info["hf_name"]
    prefix = info["subset_prefix"]

    logger.info("Loading CoIR dataset %s from %s", name, hf_name)

    # Load corpus, queries, and qrels with language-specific subsets
    corpus_ds = hf_load_dataset(hf_name, subset=f"{prefix}-corpus")
    queries_ds = hf_load_dataset(hf_name, subset=f"{prefix}-queries")
    qrels_ds = hf_load_dataset(hf_name, subset=f"{prefix}-qrels")

    # Convert corpus to our format
    corpus = _convert_corpus(corpus_ds[split], max_corpus)

    # Convert queries to our format
    queries = _convert_queries(queries_ds[split], max_queries)

    # Load qrels and add to queries
    qrels = _load_qrels(qrels_ds[split])
    _add_relevance_to_queries(queries, qrels)

    logger.info(
        "Loaded CoIR dataset %s: %d docs, %d queries",
        name,
        len(corpus),
        len(queries),
    )

    return corpus, queries


def _convert_corpus(
    hf_corpus: Any,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Convert CoIR corpus to our format."""
    corpus = []
    for i, doc in enumerate(hf_corpus):
        if max_items is not None and i >= max_items:
            break

        doc_id = doc.get("_id", str(i))
        text = doc.get("text", "")

        corpus.append({
            "doc_id": doc_id,
            "text": text,
        })

    return corpus


def _convert_queries(
    hf_queries: Any,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Convert CoIR queries to our format."""
    queries = []
    for i, q in enumerate(hf_queries):
        if max_items is not None and i >= max_items:
            break

        query_id = q.get("_id", str(i))
        text = q.get("text", "")

        queries.append({
            "query_id": query_id,
            "text": text,
            "relevant_doc_ids": [],
        })

    return queries


def _load_qrels(qrels_data: Any) -> dict[str, list[str]]:
    """Load relevance judgments from CoIR qrels.

    CoIR uses a different format: each row has query_id,
    positive_passages (list of {docid}), negative_passages.

    Returns:
        Dict mapping query_id -> list of relevant doc_ids.
    """
    qrels: dict[str, list[str]] = {}
    for row in qrels_data:
        qid = str(row.get("query_id", ""))
        positive_passages = row.get("positive_passages", [])

        relevant_docs = []
        for passage in positive_passages:
            if isinstance(passage, dict):
                doc_id = passage.get("docid", "")
            else:
                doc_id = str(passage)
            if doc_id:
                relevant_docs.append(doc_id)

        qrels[qid] = relevant_docs

    return qrels


def _add_relevance_to_queries(
    queries: list[dict[str, Any]],
    qrels: dict[str, list[str]],
) -> None:
    """Add relevant_doc_ids to queries from qrels."""
    for query in queries:
        qid = query["query_id"]
        if qid in qrels:
            query["relevant_doc_ids"] = qrels[qid]
