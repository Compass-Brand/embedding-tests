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
        texts = [doc.get("text", doc.get("title", "")) for doc in corpus]
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
        task_names: Specific task names to run.
        dry_run: If True, return empty results without running.
    """
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
        results = evaluation.run(adapter, output_folder=None)

        return {"tasks": [str(t) for t in tasks], "results": results}
    except ImportError:
        logger.warning("MTEB not installed, skipping benchmark")
        return {"tasks": [], "error": "mteb not installed"}
