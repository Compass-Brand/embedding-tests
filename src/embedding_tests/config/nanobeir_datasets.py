"""NanoBEIR dataset loader.

Loads small benchmark datasets from sentence-transformers collection.
These are fast to download and run, ideal for development and CI.
See: https://huggingface.co/datasets/sentence-transformers/NanoBEIR-en
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# NanoBEIR datasets are stored in a single HF dataset with splits
# The HF dataset is "sentence-transformers/NanoBEIR-en"
NANOBEIR_HF_DATASET = "sentence-transformers/NanoBEIR-en"

# Map our short names to HF config names (splits within the dataset)
NANOBEIR_DATASETS: dict[str, str] = {
    "nano-nfcorpus": "NanoNFCorpus",
    "nano-scifact": "NanoSciFact",
    "nano-fiqa": "NanoFiQA2018",
    "nano-arguana": "NanoArguAna",
    "nano-scidocs": "NanoSCIDOCS",
    "nano-quora": "NanoQuoraRetrieval",
}


def is_nanobeir_dataset(name: str) -> bool:
    """Check if a dataset name is a NanoBEIR dataset."""
    return name in NANOBEIR_DATASETS


def list_nanobeir_datasets() -> list[dict[str, str]]:
    """List all available NanoBEIR datasets.

    Returns:
        List of dicts with 'name' and 'hf_name' keys.
    """
    return [
        {"name": name, "hf_name": hf_name}
        for name, hf_name in NANOBEIR_DATASETS.items()
    ]


def hf_load_dataset(name: str, config_name: str | None = None, **kwargs: Any) -> Any:
    """Wrapper for HuggingFace datasets.load_dataset for easier mocking."""
    from datasets import load_dataset
    return load_dataset(name, config_name, **kwargs)


def load_nanobeir_dataset(
    name: str,
    *,
    max_corpus: int | None = None,
    max_queries: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a NanoBEIR dataset from HuggingFace.

    NanoBEIR datasets are stored in sentence-transformers/NanoBEIR-en with
    configs 'corpus', 'queries', 'qrels'. Each config contains splits named
    after the dataset (NanoNFCorpus, NanoSciFact, etc.).

    Structure:
    - load_dataset("...", "corpus")["NanoNFCorpus"] -> corpus docs
    - load_dataset("...", "queries")["NanoNFCorpus"] -> queries
    - load_dataset("...", "qrels")["NanoNFCorpus"] -> relevance judgments

    Args:
        name: NanoBEIR dataset name (e.g., "nano-nfcorpus", "nano-scifact").
        max_corpus: Maximum number of corpus documents to load.
        max_queries: Maximum number of queries to load.

    Returns:
        Tuple of (corpus, queries) in our standard format.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if name not in NANOBEIR_DATASETS:
        raise ValueError(
            f"Unknown NanoBEIR dataset: {name!r}. "
            f"Available: {', '.join(NANOBEIR_DATASETS.keys())}"
        )

    split_name = NANOBEIR_DATASETS[name]
    logger.info("Loading NanoBEIR dataset %s (split: %s)", name, split_name)

    # Load corpus, queries, and qrels from separate configs
    # Each config is a DatasetDict with splits named after each dataset
    corpus_ds = hf_load_dataset(NANOBEIR_HF_DATASET, "corpus")
    queries_ds = hf_load_dataset(NANOBEIR_HF_DATASET, "queries")
    qrels_ds = hf_load_dataset(NANOBEIR_HF_DATASET, "qrels")

    # Get the specific dataset split
    # Corpus has columns: _id, text (no title)
    # Queries has columns: _id, text
    # Qrels has columns: query-id, corpus-id, score
    corpus = _convert_corpus(corpus_ds[split_name], max_corpus)
    queries = _convert_queries(queries_ds[split_name], max_queries)

    # Load qrels and add to queries
    qrels = _load_qrels(qrels_ds.get(split_name))
    _add_relevance_to_queries(queries, qrels)

    logger.info(
        "Loaded NanoBEIR dataset %s: %d docs, %d queries",
        name,
        len(corpus),
        len(queries),
    )

    return corpus, queries


def _convert_corpus(
    hf_corpus: Any,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Convert NanoBEIR corpus to our format.

    NanoBEIR corpus has columns: id, title, text
    """
    corpus = []
    for i, doc in enumerate(hf_corpus):
        if max_items is not None and i >= max_items:
            break

        # NanoBEIR uses 'id' not '_id'
        doc_id = doc.get("id", doc.get("_id", str(i)))
        title = (doc.get("title", "") or "").strip()
        text = doc.get("text", "")

        # Combine title and text only if title is non-empty
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
    """Convert NanoBEIR queries to our format.

    NanoBEIR queries has columns: id, text
    """
    queries = []
    for i, q in enumerate(hf_queries):
        if max_items is not None and i >= max_items:
            break

        # NanoBEIR uses 'id' not '_id'
        query_id = q.get("id", q.get("_id", str(i)))
        text = q.get("text", "")

        queries.append({
            "query_id": query_id,
            "text": text,
            "relevant_doc_ids": [],
        })

    return queries


def _load_qrels(qrels_data: Any) -> dict[str, dict[str, int]]:
    """Load relevance judgments from NanoBEIR qrels.

    Returns:
        Dict mapping query_id -> {doc_id: relevance_score}
    """
    if qrels_data is None:
        return {}

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_data:
        qid = str(row.get("query-id", ""))
        did = str(row.get("corpus-id", ""))
        score = row.get("score", 1)

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
            relevant = [
                doc_id
                for doc_id, score in qrels[qid].items()
                if score > 0
            ]
            query["relevant_doc_ids"] = relevant
