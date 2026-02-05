"""RAG-specific evaluation metrics."""

from __future__ import annotations


def compute_context_recall(
    retrieved: list[str],
    relevant: list[str],
) -> float:
    """Compute context recall: fraction of relevant docs found in retrieved."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    retrieved_set = set(retrieved)
    found = len(retrieved_set & relevant_set)
    return found / len(relevant_set)


def compute_context_precision(
    retrieved: list[str],
    relevant: list[str],
) -> float:
    """Compute context precision: fraction of retrieved docs that are relevant."""
    if not retrieved:
        return 0.0
    relevant_set = set(relevant)
    retrieved_set = set(retrieved)
    found = len(retrieved_set & relevant_set)
    return found / len(retrieved_set)


def aggregate_scores(scores: list[float]) -> float:
    """Compute mean score across multiple queries."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
