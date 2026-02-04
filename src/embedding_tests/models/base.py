"""Protocol definitions for embedding and reranker models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models that encode text to vectors."""

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode texts into embedding vectors.

        Args:
            texts: List of texts to encode.
            is_query: Whether texts are queries (may add instruction prefix).
            batch_size: Batch size for encoding.

        Returns:
            Array of shape (n_texts, embedding_dim).
        """
        ...

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        ...

    def unload(self) -> None:
        """Release model resources (GPU memory, etc.)."""
        ...


@runtime_checkable
class RerankerModel(Protocol):
    """Protocol for reranker models that score query-document pairs."""

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: Candidate documents to rerank.
            top_k: Maximum number of results to return.

        Returns:
            List of (document_index, score) tuples, sorted by score descending.
        """
        ...

    def unload(self) -> None:
        """Release model resources (GPU memory, etc.)."""
        ...
