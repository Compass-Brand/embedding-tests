"""BEIR benchmark dataset loader.

Downloads and converts BEIR datasets from HuggingFace to our format.
See: https://github.com/beir-cellar/beir
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Available BEIR datasets with HuggingFace names and metadata
BEIR_DATASETS: dict[str, dict[str, Any]] = {
    "nfcorpus": {
        "hf_name": "BeIR/nfcorpus",
        "description": "Medical/nutrition information retrieval (3.6K docs, 323 queries)",
        "corpus_size": 3633,
        "query_count": 323,
    },
    "scifact": {
        "hf_name": "BeIR/scifact",
        "description": "Scientific fact verification (5K docs, 300 queries)",
        "corpus_size": 5183,
        "query_count": 300,
    },
    "fiqa": {
        "hf_name": "BeIR/fiqa",
        "description": "Financial question answering (57K docs, 648 queries)",
        "corpus_size": 57638,
        "query_count": 648,
    },
    "hotpotqa": {
        "hf_name": "BeIR/hotpotqa",
        "description": "Multi-hop question answering (5.2M docs, 7.4K queries)",
        "corpus_size": 5233329,
        "query_count": 7405,
    },
    "nq": {
        "hf_name": "BeIR/nq",
        "description": "Natural Questions - Wikipedia passages (2.6M docs, 3.4K queries)",
        "corpus_size": 2681468,
        "query_count": 3452,
    },
    "msmarco": {
        "hf_name": "BeIR/msmarco",
        "description": "MS MARCO passage ranking (8.8M docs, 6.9K queries)",
        "corpus_size": 8841823,
        "query_count": 6980,
    },
    "trec-covid": {
        "hf_name": "BeIR/trec-covid",
        "description": "COVID-19 scientific literature (171K docs, 50 queries)",
        "corpus_size": 171332,
        "query_count": 50,
    },
    "arguana": {
        "hf_name": "BeIR/arguana",
        "description": "Argument retrieval (8.6K docs, 1.4K queries)",
        "corpus_size": 8674,
        "query_count": 1406,
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

    Args:
        name: BEIR dataset name (e.g., "nfcorpus", "scifact").
        split: Dataset split to use (default: "test").
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

    # Load corpus and queries from HuggingFace
    # BEIR datasets have 'corpus' and 'queries' configs
    ds = load_dataset(hf_name)

    # Convert corpus to our format
    corpus = _convert_corpus(ds["corpus"], max_corpus)

    # Convert queries to our format
    queries = _convert_queries(ds["queries"], max_queries)

    # Load relevance judgments and add to queries
    qrels = _load_qrels(hf_name, split)
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
    """Convert HuggingFace BEIR corpus to our format."""
    corpus = []
    for i, doc in enumerate(hf_corpus):
        if max_items is not None and i >= max_items:
            break

        # BEIR format: _id, title, text
        doc_id = doc.get("_id", str(i))
        title = doc.get("title", "")
        text = doc.get("text", "")

        # Combine title and text
        full_text = f"{title}\n\n{text}".strip() if title else text

        corpus.append({
            "doc_id": doc_id,
            "text": full_text,
            "title": title,  # Keep original title for reference
        })

    return corpus


def _convert_queries(
    hf_queries: Any,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Convert HuggingFace BEIR queries to our format."""
    queries = []
    for i, q in enumerate(hf_queries):
        if max_items is not None and i >= max_items:
            break

        query_id = q.get("_id", str(i))
        text = q.get("text", "")

        queries.append({
            "query_id": query_id,
            "text": text,
            "relevant_doc_ids": [],  # Will be filled from qrels
        })

    return queries


def _load_qrels(hf_name: str, split: str) -> dict[str, dict[str, int]]:
    """Load relevance judgments (qrels) for a BEIR dataset.

    Returns:
        Dict mapping query_id -> {doc_id: relevance_score}
    """
    from datasets import load_dataset

    try:
        # BEIR qrels are stored in a separate config
        qrels_ds = load_dataset(hf_name, "qrels")
        qrels_split = qrels_ds.get(split, qrels_ds.get("test", None))

        if qrels_split is None:
            logger.warning("No qrels found for %s split %s", hf_name, split)
            return {}

        qrels: dict[str, dict[str, int]] = {}
        for row in qrels_split:
            qid = row.get("query-id", row.get("qid", ""))
            did = row.get("corpus-id", row.get("docid", ""))
            score = row.get("score", 1)

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score

        return qrels
    except Exception as e:
        logger.warning("Failed to load qrels for %s: %s", hf_name, e)
        return {}


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
