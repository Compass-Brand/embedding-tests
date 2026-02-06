"""Retrieval quality metrics: Recall@k, MRR, NDCG, Precision@k, MAP, Success@k."""

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
    """Compute Precision@k: fraction of top-k results that are relevant.

    When k > len(retrieved), only the available results are considered
    and the denominator is len(retrieved) rather than k.
    """
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


def mean_average_precision(
    queries: list[tuple[list[str], set[str]]],
) -> float:
    """Compute Mean Average Precision (MAP) across multiple queries.

    MAP is the mean of Average Precision (AP) scores across all queries.
    AP for a single query is the average of precision values at each
    position where a relevant document is found.

    Queries with empty relevant sets are excluded from the computation,
    following standard IR evaluation practice.

    Args:
        queries: List of (retrieved_doc_ids, relevant_doc_ids) tuples.

    Returns:
        MAP score in [0, 1].
    """
    if not queries:
        return 0.0

    total_ap = 0.0
    evaluated_queries = 0
    for retrieved, relevant in queries:
        if not relevant:
            continue
        evaluated_queries += 1

        ap = 0.0
        relevant_found = 0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                relevant_found += 1
                precision_at_rank = relevant_found / rank
                ap += precision_at_rank

        if relevant_found > 0:
            ap /= len(relevant)
        total_ap += ap

    return total_ap / evaluated_queries if evaluated_queries > 0 else 0.0


def success_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Compute Success@k: binary metric indicating if any relevant doc is in top-k.

    Args:
        retrieved: List of retrieved document IDs in order.
        relevant: Set of relevant document IDs.
        k: Number of top results to consider.

    Returns:
        1.0 if any relevant document is in top-k, else 0.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return 1.0 if any(doc_id in relevant for doc_id in top_k) else 0.0


def recall_at_multiple_k(
    retrieved: list[str],
    relevant: set[str],
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """Compute Recall at multiple k values.

    Args:
        retrieved: List of retrieved document IDs in order.
        relevant: Set of relevant document IDs.
        k_values: List of k values to compute. Defaults to [1, 3, 5, 10, 20].

    Returns:
        Dictionary mapping k -> recall@k.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]
    if not k_values:
        raise ValueError("k_values must not be empty")

    return {k: recall_at_k(retrieved, relevant, k) for k in k_values}


def precision_at_multiple_k(
    retrieved: list[str],
    relevant: set[str],
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """Compute Precision at multiple k values.

    Args:
        retrieved: List of retrieved document IDs in order.
        relevant: Set of relevant document IDs.
        k_values: List of k values to compute. Defaults to [1, 3, 5, 10, 20].

    Returns:
        Dictionary mapping k -> precision@k.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]
    if not k_values:
        raise ValueError("k_values must not be empty")

    return {k: precision_at_k(retrieved, relevant, k) for k in k_values}


def ndcg_at_multiple_k(
    retrieved: list[str],
    relevance: dict[str, float],
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """Compute NDCG at multiple k values.

    Args:
        retrieved: List of retrieved document IDs in order.
        relevance: Mapping of doc_id to relevance score.
        k_values: List of k values to compute. Defaults to [1, 3, 5, 10, 20].

    Returns:
        Dictionary mapping k -> ndcg@k.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]
    if not k_values:
        raise ValueError("k_values must not be empty")

    return {k: ndcg_at_k(retrieved, relevance, k) for k in k_values}


def f1_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Compute F1@k: harmonic mean of precision and recall at k.

    Args:
        retrieved: List of retrieved document IDs in order.
        relevant: Set of relevant document IDs.
        k: Number of top results to consider.

    Returns:
        F1 score in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)

    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def r_precision(
    retrieved: list[str],
    relevant: set[str],
) -> float:
    """Compute R-Precision: precision at R where R = number of relevant docs.

    R-Precision measures how well the top R results match the R relevant
    documents. It equals recall@R when exactly R documents are retrieved.

    Args:
        retrieved: List of retrieved document IDs in order.
        relevant: Set of relevant document IDs.

    Returns:
        R-Precision score in [0, 1]. Returns 0.0 if no relevant docs.
    """
    if not relevant:
        return 0.0
    r = len(relevant)
    return precision_at_k(retrieved, relevant, r)


def mean_success_at_k(
    queries: list[tuple[list[str], set[str]]],
    k: int,
) -> float:
    """Compute mean Success@k (Hit Rate) across multiple queries.

    Args:
        queries: List of (retrieved_doc_ids, relevant_doc_ids) tuples.
        k: Number of top results to consider.

    Returns:
        Mean hit rate in [0, 1].
    """
    if not queries:
        return 0.0
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    total = sum(success_at_k(retrieved, relevant, k) for retrieved, relevant in queries)
    return total / len(queries)


def compute_aggregate_stats(scores: list[float]) -> dict[str, float]:
    """Compute aggregate statistics for a list of scores.

    Args:
        scores: List of metric scores.

    Returns:
        Dictionary with mean, std, min, max, median, p25, p75 statistics.
    """
    if not scores:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    mean = sum(scores) / n

    # Standard deviation
    if n > 1:
        variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    def percentile(p: float) -> float:
        """Compute percentile using linear interpolation."""
        if n == 1:
            return sorted_scores[0]
        k = (n - 1) * p / 100
        f = int(k)
        c = min(f + 1, n - 1)
        return sorted_scores[f] + (k - f) * (sorted_scores[c] - sorted_scores[f])

    return {
        "mean": mean,
        "std": std,
        "min": sorted_scores[0],
        "max": sorted_scores[-1],
        "median": percentile(50),
        "p25": percentile(25),
        "p75": percentile(75),
    }
