"""BEIR benchmark dataset loader.

Downloads and converts BEIR datasets from HuggingFace to our format.
Uses MTEB's HuggingFace datasets (mteb/*) which include properly structured qrels.
See: https://github.com/beir-cellar/beir
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_beir_dataset(name: str) -> bool:
    """Check if a dataset name is a BEIR dataset."""
    return name in BEIR_DATASETS


def list_beir_datasets() -> list[dict[str, Any]]:
    """List all available BEIR datasets.

    Returns:
        List of dicts with dataset info.
    """
    return [
        {"name": name, **info}
        for name, info in BEIR_DATASETS.items()
    ]


# Available BEIR datasets - using MTEB's HuggingFace datasets which have qrels
# Format: mteb/{dataset_name} with 'default' config
BEIR_DATASETS: dict[str, dict[str, Any]] = {
    "nfcorpus": {
        "hf_name": "mteb/nfcorpus",
        "description": "Medical/nutrition information retrieval (3.6K docs, 323 queries)",
        "corpus_size": 3633,
        "query_count": 323,
    },
    "scifact": {
        "hf_name": "mteb/scifact",
        "description": "Scientific fact verification (5K docs, 300 queries)",
        "corpus_size": 5183,
        "query_count": 300,
    },
    "fiqa": {
        "hf_name": "mteb/fiqa",
        "description": "Financial question answering (57K docs, 648 queries)",
        "corpus_size": 57638,
        "query_count": 648,
    },
    "hotpotqa": {
        "hf_name": "mteb/hotpotqa",
        "description": "Multi-hop question answering (5.2M docs, 7.4K queries)",
        "corpus_size": 5233329,
        "query_count": 7405,
    },
    "nq": {
        "hf_name": "mteb/nq",
        "description": "Natural Questions - Wikipedia passages (2.6M docs, 3.4K queries)",
        "corpus_size": 2681468,
        "query_count": 3452,
    },
    "msmarco": {
        "hf_name": "mteb/msmarco",
        "description": "MS MARCO passage ranking (8.8M docs, 6.9K queries)",
        "corpus_size": 8841823,
        "query_count": 6980,
    },
    "trec-covid": {
        "hf_name": "mteb/trec-covid",
        "description": "COVID-19 scientific literature (171K docs, 50 queries)",
        "corpus_size": 171332,
        "query_count": 50,
    },
    "arguana": {
        "hf_name": "mteb/arguana",
        "description": "Argument retrieval (8.6K docs, 1.4K queries)",
        "corpus_size": 8674,
        "query_count": 1406,
    },
    "scidocs": {
        "hf_name": "mteb/scidocs",
        "description": "Scientific document retrieval (25K docs, 1K queries)",
        "corpus_size": 25657,
        "query_count": 1000,
    },
}


def load_beir_dataset(
    name: str,
    *,
    split: str = "test",
    max_corpus: int | None = None,
    max_queries: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a BEIR dataset from HuggingFace and convert to our format.

    Uses MTEB's HuggingFace datasets (mteb/*) which store data with
    separate configs for 'corpus', 'queries', and 'default' (qrels).

    Structure:
    - load_dataset("mteb/nfcorpus", "corpus")["train"] -> corpus docs
    - load_dataset("mteb/nfcorpus", "queries")["queries"] -> queries
    - load_dataset("mteb/nfcorpus", "default")[split] -> qrels

    Args:
        name: BEIR dataset name (e.g., "nfcorpus", "scifact").
        split: Dataset split for qrels (default: "test").
        max_corpus: Maximum number of corpus documents to load (for testing).
        max_queries: Maximum number of queries to load (for testing).

    Returns:
        Tuple of (corpus, queries) in our standard format.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if name not in BEIR_DATASETS:
        raise ValueError(
            f"Unknown BEIR dataset: {name!r}. "
            f"Available: {', '.join(BEIR_DATASETS.keys())}"
        )

    from datasets import load_dataset

    info = BEIR_DATASETS[name]
    hf_name = info["hf_name"]

    logger.info("Loading BEIR dataset %s from %s", name, hf_name)

    # Load corpus - usually has 'train' split containing all docs
    corpus_ds = load_dataset(hf_name, "corpus")
    corpus_split = corpus_ds.get("train", corpus_ds.get(list(corpus_ds.keys())[0]))

    # Load queries - usually has 'queries' split
    queries_ds = load_dataset(hf_name, "queries")
    queries_split = queries_ds.get("queries", queries_ds.get(list(queries_ds.keys())[0]))

    # Load qrels (default config has train/dev/test splits)
    qrels_ds = load_dataset(hf_name, "default")
    qrels_split = qrels_ds.get(split)
    if qrels_split is None:
        available_splits = list(qrels_ds.keys())
        raise ValueError(f"Split '{split}' not found. Available: {available_splits}")

    # Convert corpus and queries to our format
    corpus = _convert_corpus(corpus_split, max_corpus)
    queries = _convert_queries(queries_split, max_queries)

    # Build qrels dict from qrels split (columns: query-id, corpus-id, score)
    qrels = _build_qrels_from_dataset(qrels_split)
    _add_relevance_to_queries(queries, qrels)

    logger.info(
        "Loaded BEIR dataset %s: %d docs, %d queries",
        name,
        len(corpus),
        len(queries),
    )

    return corpus, queries


def _convert_corpus(
    hf_corpus: Any,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Convert MTEB BEIR corpus to our format.

    MTEB corpus can be:
    1. A HuggingFace Dataset with 'id', 'title', 'text' columns
    2. A list/iterable of dicts
    """
    corpus = []

    # Handle HuggingFace Dataset or list
    for i, doc in enumerate(hf_corpus):
        if max_items is not None and i >= max_items:
            break

        # MTEB uses 'id', older BEIR uses '_id'
        doc_id = doc.get("id", doc.get("_id", str(i)))
        title = (doc.get("title", "") or "").strip()
        text = doc.get("text", "")

        # Combine title and text (only if title is non-empty after stripping)
        full_text = f"{title}\n\n{text}".strip() if title else text

        corpus.append({
            "doc_id": doc_id,
            "text": full_text,
            "title": title,
        })

    return corpus


def _convert_queries(
    hf_queries: Any,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Convert MTEB BEIR queries to our format.

    MTEB queries can be:
    1. A HuggingFace Dataset with 'id', 'text' columns
    2. A list/iterable of dicts
    """
    queries = []

    for i, q in enumerate(hf_queries):
        if max_items is not None and i >= max_items:
            break

        # MTEB uses 'id', older BEIR uses '_id'
        query_id = q.get("id", q.get("_id", str(i)))
        text = q.get("text", "")

        queries.append({
            "query_id": query_id,
            "text": text,
            "relevant_doc_ids": [],
        })

    return queries


def _build_qrels_from_dataset(qrels_ds: Any) -> dict[str, dict[str, int]]:
    """Build qrels dict from MTEB qrels dataset.

    MTEB qrels datasets have columns: query-id, corpus-id, score

    Returns:
        Dict mapping query_id -> {doc_id: relevance_score}
    """
    if qrels_ds is None:
        return {}

    qrels: dict[str, dict[str, int]] = {}

    for row in qrels_ds:
        qid = str(row.get("query-id", ""))
        did = str(row.get("corpus-id", ""))
        score = int(row.get("score", 1))

        if qid and did:
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score

    return qrels


def _add_relevance_to_queries(
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
) -> None:
    """Add relevant_doc_ids to queries from qrels."""
    for query in queries:
        qid = query["query_id"]
        if qid in qrels:
            # Get all docs with positive relevance score
            relevant = [
                doc_id
                for doc_id, score in qrels[qid].items()
                if score > 0
            ]
            query["relevant_doc_ids"] = relevant
