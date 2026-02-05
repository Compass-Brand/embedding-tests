"""Vector retrieval with ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
import numpy as np


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    doc_id: str
    score: float


class VectorStore:
    """In-memory vector store backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str = "default",
        embedding_dim: int = 768,
        metric: str = "cosine",
    ) -> None:
        self._client = chromadb.Client()
        metadata = {"hnsw:space": metric}
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
        )
        self._embedding_dim = embedding_dim

    def index(self, embeddings: np.ndarray, doc_ids: list[str]) -> None:
        """Add embeddings to the store."""
        self._collection.add(
            embeddings=embeddings.tolist(),
            ids=doc_ids,
        )

    def query(self, query_embedding: np.ndarray, top_k: int = 10) -> list[RetrievalResult]:
        """Query for similar documents."""
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        output: list[RetrievalResult] = []
        if results["ids"] and results["distances"]:
            for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
                # ChromaDB cosine distance ranges from 0 to 2; convert to similarity
                score = 1.0 - (distance / 2.0)
                output.append(RetrievalResult(doc_id=doc_id, score=score))

        # Sort by score descending
        output.sort(key=lambda r: r.score, reverse=True)
        return output

    def count(self) -> int:
        """Return number of documents in store."""
        return self._collection.count()

    def clear(self) -> None:
        """Remove all documents from the store."""
        name = self._collection.name
        metadata = self._collection.metadata
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(name=name, metadata=metadata)
