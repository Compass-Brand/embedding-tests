"""Two-stage retrieval with reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embedding_tests.models.base import RerankerModel


@dataclass
class RerankResult:
    """A reranked document result."""

    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def rerank_results(
    query: str,
    documents: list[dict[str, Any]],
    reranker: RerankerModel,
    *,
    top_k: int = 10,
) -> list[RerankResult]:
    """Rerank retrieved documents using a cross-encoder reranker."""
    doc_texts = [d["text"] for d in documents]
    ranked = reranker.rerank(query, doc_texts, top_k=top_k)

    results: list[RerankResult] = []
    for idx, score in ranked:
        doc = documents[idx]
        metadata = {k: v for k, v in doc.items() if k not in ("doc_id", "text")}
        results.append(
            RerankResult(
                doc_id=doc["doc_id"],
                score=score,
                text=doc["text"],
                metadata=metadata,
            )
        )

    return results
