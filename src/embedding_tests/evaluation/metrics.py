"""Retrieval quality metrics: Recall@k, MRR, NDCG, Precision@k."""

from __future__ import annotations

import math


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Compute Recall@k: fraction of relevant docs in top-k results."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    found = sum(1 for doc_id in top_k if doc_id in relevant)
    return found / len(relevant)


def precision_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Compute Precision@k: fraction of top-k results that are relevant."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    found = sum(1 for doc_id in top_k if doc_id in relevant)
    return found / len(top_k)


def mrr(
    queries: list[tuple[list[str], set[str]]],
) -> float:
    """Compute Mean Reciprocal Rank across multiple queries.

    Args:
        queries: List of (retrieved_doc_ids, relevant_doc_ids) tuples.
    """
    if not queries:
        return 0.0
    total = 0.0
    for retrieved, relevant in queries:
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                total += 1.0 / rank
                break
    return total / len(queries)


def ndcg_at_k(
    retrieved: list[str],
    relevance: dict[str, float],
    k: int,
) -> float:
    """Compute NDCG@k for a single query.

    Args:
        retrieved: List of retrieved document IDs in order.
        relevance: Mapping of doc_id to relevance score.
        k: Number of results to consider.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    top_k = retrieved[:k]

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        rel = relevance.get(doc_id, 0.0)
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
