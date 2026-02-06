"""MTEB retrieval task dataset loader.

Loads datasets using MTEB's task infrastructure for retrieval benchmarks.
See: https://github.com/embeddings-benchmark/mteb
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Map short names to MTEB task names
MTEB_RETRIEVAL_TASKS: dict[str, str] = {
    # CQADupStack suite (technical Q&A)
    "cqadupstack-android": "CQADupstackAndroidRetrieval",
    "cqadupstack-english": "CQADupstackEnglishRetrieval",
    "cqadupstack-gaming": "CQADupstackGamingRetrieval",
    "cqadupstack-gis": "CQADupstackGisRetrieval",
    "cqadupstack-mathematica": "CQADupstackMathematicaRetrieval",
    "cqadupstack-physics": "CQADupstackPhysicsRetrieval",
    "cqadupstack-programmers": "CQADupstackProgrammersRetrieval",
    "cqadupstack-stats": "CQADupstackStatsRetrieval",
    "cqadupstack-tex": "CQADupstackTexRetrieval",
    "cqadupstack-unix": "CQADupstackUnixRetrieval",
    "cqadupstack-webmasters": "CQADupstackWebmastersRetrieval",
    "cqadupstack-wordpress": "CQADupstackWordpressRetrieval",
    # Other MTEB retrieval tasks
    "stackoverflow-dupquestions": "StackOverflowDupQuestions",
}


def is_mteb_dataset(name: str) -> bool:
    """Check if a dataset name is an MTEB retrieval task."""
    return name in MTEB_RETRIEVAL_TASKS


def list_mteb_datasets() -> list[dict[str, str]]:
    """List all available MTEB retrieval datasets.

    Returns:
        List of dicts with 'name' and 'task_name' keys.
    """
    return [
        {"name": name, "task_name": task_name}
        for name, task_name in MTEB_RETRIEVAL_TASKS.items()
    ]


def load_mteb_dataset(
    name: str,
    *,
    split: str = "test",
    max_corpus: int | None = None,
    max_queries: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load an MTEB retrieval dataset.

    Uses MTEB's task infrastructure to load corpus, queries, and qrels.

    Args:
        name: MTEB dataset name (e.g., "cqadupstack-programmers").
        split: Dataset split to use (default: "test").
        max_corpus: Maximum number of corpus documents to load.
        max_queries: Maximum number of queries to load.

    Returns:
        Tuple of (corpus, queries) in our standard format.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if name not in MTEB_RETRIEVAL_TASKS:
        raise ValueError(
            f"Unknown MTEB retrieval task: {name!r}. "
            f"Available: {', '.join(MTEB_RETRIEVAL_TASKS.keys())}"
        )

    task_name = MTEB_RETRIEVAL_TASKS[name]
    logger.info("Loading MTEB task %s (%s)", name, task_name)

    import mteb

    tasks = mteb.get_tasks(tasks=[task_name])
    if not tasks:
        raise ValueError(f"MTEB task not found: {task_name}")

    task = tasks[0]
    task.load_data(eval_splits=[split])

    # Convert corpus to our format
    corpus = _convert_corpus(task.corpus.get(split, {}), max_corpus)

    # Convert queries to our format
    queries = _convert_queries(task.queries.get(split, {}), max_queries)

    # Load qrels and add to queries
    qrels = task.relevant_docs.get(split, {})
    _add_relevance_to_queries(queries, qrels)

    logger.info(
        "Loaded MTEB task %s: %d docs, %d queries",
        name,
        len(corpus),
        len(queries),
    )

    return corpus, queries


def _convert_corpus(
    mteb_corpus: dict[str, Any],
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Convert MTEB corpus to our format.

    MTEB corpus is a dict mapping doc_id -> doc_info where doc_info
    can be a dict with 'text' and 'title' or just a string.
    """
    corpus = []
    for i, (doc_id, doc_info) in enumerate(mteb_corpus.items()):
        if max_items is not None and i >= max_items:
            break

        # Handle both dict and string corpus entries
        if isinstance(doc_info, dict):
            text = doc_info.get("text", "")
            title = (doc_info.get("title", "") or "").strip()
            # Combine title and text only if title is non-empty
            full_text = f"{title}\n\n{text}".strip() if title else text
        else:
            full_text = str(doc_info)
            title = ""

        corpus.append({
            "doc_id": doc_id,
            "text": full_text,
            "title": title,
        })

    return corpus


def _convert_queries(
    mteb_queries: dict[str, str],
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Convert MTEB queries to our format.

    MTEB queries is a dict mapping query_id -> query_text.
    """
    queries = []
    for i, (query_id, query_text) in enumerate(mteb_queries.items()):
        if max_items is not None and i >= max_items:
            break

        queries.append({
            "query_id": query_id,
            "text": query_text,
            "relevant_doc_ids": [],
        })

    return queries


def _add_relevance_to_queries(
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
) -> None:
    """Add relevant_doc_ids to queries from qrels.

    MTEB qrels is a dict mapping query_id -> {doc_id: relevance_score}.
    """
    for query in queries:
        qid = query["query_id"]
        if qid in qrels:
            relevant = [
                doc_id
                for doc_id, score in qrels[qid].items()
                if score > 0
            ]
            query["relevant_doc_ids"] = relevant
