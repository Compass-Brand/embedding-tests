"""MTEB benchmark integration."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from embedding_tests.models.base import EmbeddingModel

logger = logging.getLogger(__name__)

# MTEB task types for filtering benchmarks
MTEB_TASK_TYPES: list[str] = [
    "BitextMining",
    "Classification",
    "Clustering",
    "MultilabelClassification",
    "PairClassification",
    "Reranking",
    "Retrieval",
    "STS",
    "Summarization",
]

# Recommended small retrieval tasks for quick evaluation
# These are fast to run and provide good signal for embedding quality
RECOMMENDED_RETRIEVAL_TASKS: list[str] = [
    "NFCorpus",  # Medical/nutrition (3.6K docs, 323 queries)
    "SciFact",  # Scientific fact verification (5K docs, 300 queries)
    "FiQA2018",  # Financial QA (57K docs, 648 queries)
    "ArguAna",  # Argument retrieval (8.6K docs, 1.4K queries)
]


class MTEBModelAdapter:
    """Adapts our EmbeddingModel to the MTEB model interface."""

    def __init__(self, model: EmbeddingModel) -> None:
        self._model = model

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode sentences (document mode)."""
        return self._model.encode(sentences, is_query=False, batch_size=batch_size)

    def encode_queries(
        self,
        queries: list[str],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode queries (query mode with instruction)."""
        return self._model.encode(queries, is_query=True, batch_size=batch_size)

    def encode_corpus(
        self,
        corpus: list[dict[str, str]],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode corpus documents."""
        texts = []
        for i, doc in enumerate(corpus):
            text = doc.get("text") or doc.get("title") or ""
            if not text:
                logger.warning("Corpus document %d has no 'text' or 'title' field", i)
            texts.append(text)
        return self._model.encode(texts, is_query=False, batch_size=batch_size)


def run_mteb_tasks(
    model: EmbeddingModel,
    *,
    task_types: list[str] | None = None,
    task_names: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run MTEB benchmark tasks.

    Args:
        model: The embedding model to evaluate.
        task_types: Filter by task type (e.g., "Retrieval", "Reranking").
            Exactly one of task_types or task_names must be provided.
        task_names: Specific task names to run.
            Exactly one of task_types or task_names must be provided.
        dry_run: If True, return empty results without running.

    Raises:
        ValueError: If both task_types and task_names are provided.
    """
    if task_types is not None and task_names is not None:
        raise ValueError("Provide only one of task_types or task_names")
    if task_types is None and task_names is None:
        raise ValueError("Either task_types or task_names must be provided")
    if (task_types is not None and not task_types) or (task_names is not None and not task_names):
        raise ValueError("task_types and task_names must not be empty")

    if dry_run:
        return {"tasks": [], "dry_run": True}

    try:
        import mteb

        adapter = MTEBModelAdapter(model)

        tasks = []
        if task_names:
            tasks = mteb.get_tasks(tasks=task_names)
        elif task_types:
            tasks = mteb.get_tasks(task_types=task_types)

        evaluation = mteb.MTEB(tasks=tasks)
        # adapter wraps our model to satisfy MTEB's expected interface
        results = evaluation.run(adapter, output_folder=None)

        return {"tasks": [str(t) for t in tasks], "results": results}
    except ImportError:
        logger.warning("MTEB not installed, skipping benchmark")
        return {"tasks": [], "error": "mteb not installed"}


def format_mteb_results(raw_results: list[Any]) -> dict[str, dict[str, float]]:
    """Format MTEB results into a structured dictionary.

    Args:
        raw_results: List of MTEB TaskResult objects.

    Returns:
        Dict mapping task_name -> {metric_name: score}.
        Extracts common metrics like ndcg_at_10, mrr_at_10, etc.
    """
    if not raw_results:
        return {}

    formatted: dict[str, dict[str, float]] = {}

    for result in raw_results:
        task_name = result.task_name
        scores = result.scores

        # MTEB scores are nested: {"test": [{metric: value, ...}]}
        # We prefer the "test" split; fall back to first available split.
        # For multilingual tasks, each split may have multiple score dicts
        # (per-language), but we only extract the first (aggregate).
        metrics: dict[str, float] = {}

        # Prefer "test" split if present; fall back to first available if absent.
        # Note: we check for key presence, not falsiness - an empty test split
        # is treated differently than a missing one.
        split_scores = scores.get("test") if "test" in scores else next(
            iter(scores.values()), None
        )
        if split_scores and isinstance(split_scores, list):
            score_dict = split_scores[0]
            if not isinstance(score_dict, dict):
                continue
            for metric_name, value in score_dict.items():
                if isinstance(value, (int, float)):
                    metrics[metric_name] = float(value)

        if metrics:
            formatted[task_name] = metrics

    return formatted
