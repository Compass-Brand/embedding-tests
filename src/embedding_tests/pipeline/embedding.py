"""Batch embedding pipeline with memory management."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from embedding_tests.models.base import EmbeddingModel


@dataclass
class EmbeddingResult:
    """Result of a batch embedding operation."""

    embeddings: np.ndarray
    total_time_seconds: float
    num_texts: int
    batch_size: int


def batch_embed(
    model: EmbeddingModel,
    texts: list[str],
    *,
    batch_size: int = 32,
    is_query: bool = False,
) -> EmbeddingResult:
    """Embed texts in batches with timing."""
    start = time.perf_counter()
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Outer loop handles memory-controlled batching; model uses its own internal batching
        embeddings = model.encode(batch, is_query=is_query)
        all_embeddings.append(embeddings)

    elapsed = time.perf_counter() - start
    if all_embeddings:
        combined = np.concatenate(all_embeddings, axis=0)
    else:
        # TODO: Consider inferring dtype from model output for consistency
        combined = np.empty((0, model.get_embedding_dim()), dtype=np.float32)

    return EmbeddingResult(
        embeddings=combined,
        total_time_seconds=elapsed,
        num_texts=len(texts),
        batch_size=batch_size,
    )
