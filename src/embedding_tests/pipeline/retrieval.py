"""Vector retrieval with ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
import chromadb.errors
import numpy as np


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    doc_id: str
    score: float


class VectorStore:
    """In-memory vector store backed by ChromaDB."""

    _VALID_METRICS = {"cosine", "l2", "ip"}

    def __init__(
        self,
        collection_name: str = "default",
        embedding_dim: int = 768,
        metric: str = "cosine",
    ) -> None:
        if metric not in self._VALID_METRICS:
            raise ValueError(
                f"Invalid metric: {metric!r}. Must be one of {sorted(self._VALID_METRICS)}"
            )
        self._client = chromadb.Client()
        metadata = {"hnsw:space": metric}
        # Delete existing collection to ensure metric consistency
        try:
            self._client.delete_collection(name=collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass  # Collection doesn't exist
        self._collection = self._client.create_collection(
            name=collection_name,
            metadata=metadata,
        )
        self._embedding_dim = embedding_dim
        self._metric = metric

    def index(
        self, embeddings: np.ndarray, doc_ids: list[str], batch_size: int = 5000
    ) -> None:
        """Add embeddings to the store.

        Args:
            embeddings: Embedding vectors as numpy array.
            doc_ids: List of document IDs corresponding to embeddings.
            batch_size: Maximum batch size for ChromaDB inserts (default 5000).
        """
        if embeddings.shape[0] != len(doc_ids):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) must match "
                f"number of doc_ids ({len(doc_ids)})"
            )
        if embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        # Batch inserts to avoid ChromaDB max batch size limit
        n_total = embeddings.shape[0]
        for start_idx in range(0, n_total, batch_size):
            end_idx = min(start_idx + batch_size, n_total)
            batch_embeddings = embeddings[start_idx:end_idx]
            batch_ids = doc_ids[start_idx:end_idx]
            self._collection.add(
                embeddings=batch_embeddings.tolist(),
                ids=batch_ids,
            )

    def query(self, query_embedding: np.ndarray, top_k: int = 10) -> list[RetrievalResult]:
        """Query for similar documents."""
        if query_embedding.shape[0] != self._embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self._embedding_dim}, "
                f"got {query_embedding.shape[0]}"
            )
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        output: list[RetrievalResult] = []
        if results["ids"] and results["distances"]:
            for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
                score = self._distance_to_score(distance)
                output.append(RetrievalResult(doc_id=doc_id, score=score))

        # Sort by score descending
        output.sort(key=lambda r: r.score, reverse=True)
        return output

    def _distance_to_score(self, distance: float) -> float:
        """Convert a distance value to a similarity score based on the metric."""
        if self._metric == "cosine":
            return 1.0 - (distance / 2.0)
        elif self._metric == "l2":
            return 1.0 / (1.0 + distance)
        elif self._metric == "ip":
            dot = 1.0 - distance
            return max(0.0, min(1.0, (dot + 1.0) / 2.0))
        raise AssertionError(f"Unexpected metric: {self._metric}")

    def count(self) -> int:
        """Return number of documents in store."""
        return self._collection.count()

    def clear(self) -> None:
        """Remove all documents from the store."""
        name = self._collection.name
        metadata = self._collection.metadata
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(name=name, metadata=metadata)
