"""RAG-specific evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Available RAGAS metrics
RAGAS_METRICS: list[str] = [
    "faithfulness",
    "answer_relevance",
    "context_recall",
    "context_precision",
    "context_relevancy",
    "answer_correctness",
]


@dataclass
class RAGEvaluationSample:
    """A single sample for RAG evaluation.

    Attributes:
        question: The user query/question.
        contexts: Retrieved text contexts (for LLM-based metrics like faithfulness).
        answer: The generated answer.
        ground_truth: Expected answer (for answer correctness metrics).
        relevant_doc_ids: Set of ground-truth relevant document IDs.
        retrieved_doc_ids: IDs of retrieved documents (for retrieval metrics).
            If not provided, falls back to using contexts as doc IDs.
    """

    question: str
    contexts: list[str]
    answer: str
    ground_truth: str | None = None
    relevant_doc_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)


class RAGEvaluator:
    """Evaluator for RAG pipeline quality metrics."""

    def __init__(self, llm: Any = None) -> None:
        """Initialize the evaluator.

        Args:
            llm: Optional LLM for computing faithfulness/answer relevance.
                 If not provided, LLM-based metrics are skipped.
        """
        self._llm = llm

    def evaluate(
        self,
        samples: list[RAGEvaluationSample],
        *,
        metrics: list[str] | None = None,
        return_per_sample: bool = False,
    ) -> dict[str, Any]:
        """Evaluate RAG samples on specified metrics.

        Args:
            samples: List of RAG evaluation samples.
            metrics: Which metrics to compute. Defaults to context-based metrics.
            return_per_sample: If True, include per-sample breakdown.

        Returns:
            Dict with aggregate scores and optionally per-sample scores.
        """
        if metrics is None:
            metrics = ["context_recall", "context_precision"]

        if not samples:
            result: dict[str, Any] = {}
            for metric in metrics:
                if metric in ("context_recall", "context_precision"):
                    result[metric] = 0.0
            return result

        # Compute per-sample scores
        per_sample_scores: list[dict[str, float]] = []
        for sample in samples:
            sample_scores: dict[str, float] = {}
            # Use retrieved_doc_ids for retrieval metrics if available,
            # otherwise fall back to contexts (for backward compatibility)
            retrieved = sample.retrieved_doc_ids or sample.contexts
            if "context_recall" in metrics:
                sample_scores["context_recall"] = compute_context_recall(
                    retrieved, sample.relevant_doc_ids
                )
            if "context_precision" in metrics:
                sample_scores["context_precision"] = compute_context_precision(
                    retrieved, sample.relevant_doc_ids
                )
            per_sample_scores.append(sample_scores)

        # Aggregate scores
        result = {}
        for metric in metrics:
            # Skip LLM-based metrics if no LLM configured
            if metric in ("faithfulness", "answer_relevance") and self._llm is None:
                continue
            if metric in ("context_recall", "context_precision"):
                scores = [s.get(metric, 0.0) for s in per_sample_scores]
                result[metric] = aggregate_scores(scores)

        if return_per_sample:
            result["per_sample"] = per_sample_scores

        return result


def compute_context_recall(
    retrieved: list[str],
    relevant: list[str],
) -> float:
    """Compute context recall: fraction of relevant docs found in retrieved.

    Returns 0.0 when ``relevant`` is empty, since there are no relevant
    documents to recall.
    """
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
    """Compute context precision: fraction of retrieved docs that are relevant.

    Counts matches against the raw retrieved list (preserving duplicates)
    so that duplicate retrievals reduce precision appropriately.
    """
    if not retrieved:
        return 0.0
    relevant_set = set(relevant)
    found = sum(1 for doc in retrieved if doc in relevant_set)
    return found / len(retrieved)


def aggregate_scores(scores: list[float]) -> float:
    """Compute mean score across multiple queries."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
