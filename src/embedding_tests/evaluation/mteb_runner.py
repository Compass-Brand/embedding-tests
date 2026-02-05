"""MTEB benchmark integration."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from embedding_tests.models.base import EmbeddingModel

logger = logging.getLogger(__name__)


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
            text = doc.get("text") if doc.get("text") is not None else doc.get("title") or ""
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
