"""Tests for RAG evaluator."""

from __future__ import annotations

import pytest

from embedding_tests.evaluation.rag_evaluator import (
    compute_context_recall,
    compute_context_precision,
    aggregate_scores,
)


class TestContextRecall:
    """Tests for context recall computation."""

    def test_context_recall_perfect(self) -> None:
        retrieved = ["d1", "d2"]
        relevant = ["d1", "d2"]
        assert compute_context_recall(retrieved, relevant) == 1.0

    def test_context_recall_none(self) -> None:
        retrieved = ["d3", "d4"]
        relevant = ["d1", "d2"]
        assert compute_context_recall(retrieved, relevant) == 0.0

    def test_context_recall_partial(self) -> None:
        retrieved = ["d1", "d3"]
        relevant = ["d1", "d2"]
        assert compute_context_recall(retrieved, relevant) == pytest.approx(0.5)

    def test_context_recall_empty_relevant(self) -> None:
        retrieved = ["d1", "d2"]
        relevant: list[str] = []
        assert compute_context_recall(retrieved, relevant) == 0.0

    def test_context_recall_both_empty(self) -> None:
        assert compute_context_recall([], []) == 0.0

    def test_context_recall_with_duplicates(self) -> None:
        """Verify duplicates in retrieved are deduplicated via set conversion."""
        retrieved = ["d1", "d1", "d2"]
        relevant = ["d1", "d2"]
        # set(retrieved) = {"d1", "d2"}, intersection = {"d1", "d2"} -> 2/2 = 1.0
        assert compute_context_recall(retrieved, relevant) == 1.0


class TestContextPrecision:
    """Tests for context precision computation."""

    def test_context_precision_perfect(self) -> None:
        retrieved = ["d1", "d2"]
        relevant = ["d1", "d2"]
        assert compute_context_precision(retrieved, relevant) == 1.0

    def test_context_precision_half(self) -> None:
        retrieved = ["d1", "d3"]
        relevant = ["d1", "d2"]
        assert compute_context_precision(retrieved, relevant) == pytest.approx(0.5)

    def test_context_precision_empty_retrieved(self) -> None:
        retrieved: list[str] = []
        relevant = ["d1", "d2"]
        assert compute_context_precision(retrieved, relevant) == 0.0

    def test_context_precision_none_relevant(self) -> None:
        retrieved = ["d3", "d4"]
        relevant = ["d1", "d2"]
        assert compute_context_precision(retrieved, relevant) == 0.0

    def test_context_precision_with_duplicates(self) -> None:
        """Verify duplicates in retrieved count individually against relevant set."""
        retrieved = ["d1", "d1", "d3"]
        relevant = ["d1", "d2"]
        # "d1" matches twice, "d3" does not -> found=2, len(retrieved)=3 -> 2/3
        assert compute_context_precision(retrieved, relevant) == pytest.approx(2.0 / 3.0)


class TestAggregateScores:
    """Tests for score aggregation."""

    def test_aggregate_across_queries(self) -> None:
        scores = [0.5, 1.0, 0.0]
        assert aggregate_scores(scores) == pytest.approx(0.5)

    def test_aggregate_empty(self) -> None:
        assert aggregate_scores([]) == 0.0


class TestRAGEvaluationSample:
    """Tests for RAG evaluation sample dataclass."""

    def test_rag_sample_creation(self) -> None:
        """Should create RAG evaluation sample with required fields."""
        from embedding_tests.evaluation.rag_evaluator import RAGEvaluationSample

        sample = RAGEvaluationSample(
            question="What is the capital of France?",
            contexts=["Paris is the capital of France."],
            answer="Paris",
            ground_truth="Paris",
        )
        assert sample.question == "What is the capital of France?"
        assert sample.contexts == ["Paris is the capital of France."]
        assert sample.answer == "Paris"
        assert sample.ground_truth == "Paris"

    def test_rag_sample_optional_ground_truth(self) -> None:
        """Should allow ground_truth to be optional."""
        from embedding_tests.evaluation.rag_evaluator import RAGEvaluationSample

        sample = RAGEvaluationSample(
            question="What is the capital of France?",
            contexts=["Paris is the capital of France."],
            answer="Paris",
        )
        assert sample.ground_truth is None

    def test_rag_sample_relevant_doc_ids(self) -> None:
        """Should support relevant_doc_ids for retrieval evaluation."""
        from embedding_tests.evaluation.rag_evaluator import RAGEvaluationSample

        sample = RAGEvaluationSample(
            question="Q1",
            contexts=["Context text 1", "Context text 2"],
            answer="A1",
            relevant_doc_ids=["doc1", "doc3"],
        )
        assert sample.relevant_doc_ids == ["doc1", "doc3"]

    def test_rag_sample_retrieved_doc_ids(self) -> None:
        """Should support retrieved_doc_ids separate from contexts."""
        from embedding_tests.evaluation.rag_evaluator import RAGEvaluationSample

        sample = RAGEvaluationSample(
            question="Q1",
            contexts=["Context text 1", "Context text 2"],
            answer="A1",
            relevant_doc_ids=["doc1", "doc3"],
            retrieved_doc_ids=["doc1", "doc2"],
        )
        assert sample.retrieved_doc_ids == ["doc1", "doc2"]
        assert sample.contexts == ["Context text 1", "Context text 2"]

    def test_rag_sample_default_doc_ids(self) -> None:
        """Should default doc_ids to empty lists."""
        from embedding_tests.evaluation.rag_evaluator import RAGEvaluationSample

        sample = RAGEvaluationSample(
            question="Q1",
            contexts=["Context"],
            answer="A1",
        )
        assert sample.relevant_doc_ids == []
        assert sample.retrieved_doc_ids == []


class TestRAGEvaluator:
    """Tests for the main RAG evaluator class."""

    def test_evaluator_computes_context_metrics(self) -> None:
        """Evaluator should compute context recall and precision."""
        from embedding_tests.evaluation.rag_evaluator import (
            RAGEvaluator,
            RAGEvaluationSample,
        )

        samples = [
            RAGEvaluationSample(
                question="What is X?",
                contexts=["Context text 1", "Context text 2"],
                answer="Answer",
                relevant_doc_ids=["doc1", "doc3"],
                retrieved_doc_ids=["doc1", "doc2"],
            )
        ]
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(samples)

        assert "context_recall" in results
        assert "context_precision" in results
        # doc1 found out of [doc1, doc3] = 0.5 recall
        assert results["context_recall"] == pytest.approx(0.5)
        # doc1 is relevant out of [doc1, doc2] = 0.5 precision
        assert results["context_precision"] == pytest.approx(0.5)

    def test_evaluator_aggregates_across_samples(self) -> None:
        """Evaluator should aggregate metrics across multiple samples."""
        from embedding_tests.evaluation.rag_evaluator import (
            RAGEvaluator,
            RAGEvaluationSample,
        )

        samples = [
            RAGEvaluationSample(
                question="Q1",
                contexts=["Context 1"],
                answer="A1",
                relevant_doc_ids=["doc1"],
                retrieved_doc_ids=["doc1"],
            ),
            RAGEvaluationSample(
                question="Q2",
                contexts=["Context 2", "Context 3"],
                answer="A2",
                relevant_doc_ids=["doc2"],
                retrieved_doc_ids=["doc2", "doc3"],
            ),
        ]
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(samples)

        # Mean recall: (1.0 + 1.0) / 2 = 1.0
        assert results["context_recall"] == pytest.approx(1.0)
        # Mean precision: (1.0 + 0.5) / 2 = 0.75
        assert results["context_precision"] == pytest.approx(0.75)

    def test_evaluator_handles_empty_samples(self) -> None:
        """Evaluator should handle empty sample list gracefully."""
        from embedding_tests.evaluation.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        results = evaluator.evaluate([])

        assert results["context_recall"] == 0.0
        assert results["context_precision"] == 0.0

    def test_evaluator_returns_per_sample_scores(self) -> None:
        """Evaluator should optionally return per-sample breakdown."""
        from embedding_tests.evaluation.rag_evaluator import (
            RAGEvaluator,
            RAGEvaluationSample,
        )

        samples = [
            RAGEvaluationSample(
                question="Q1",
                contexts=["Context 1"],
                answer="A1",
                relevant_doc_ids=["doc1"],
                retrieved_doc_ids=["doc1"],
            ),
            RAGEvaluationSample(
                question="Q2",
                contexts=["Context 2"],
                answer="A2",
                relevant_doc_ids=["doc2"],
                retrieved_doc_ids=["doc3"],
            ),
        ]
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(samples, return_per_sample=True)

        assert "per_sample" in results
        assert len(results["per_sample"]) == 2
        assert results["per_sample"][0]["context_recall"] == 1.0
        assert results["per_sample"][1]["context_recall"] == 0.0

    def test_evaluator_select_metrics(self) -> None:
        """Should allow selecting which metrics to compute."""
        from embedding_tests.evaluation.rag_evaluator import (
            RAGEvaluator,
            RAGEvaluationSample,
        )

        samples = [
            RAGEvaluationSample(
                question="Q1",
                contexts=["Context 1"],
                answer="A1",
                relevant_doc_ids=["doc1"],
                retrieved_doc_ids=["doc1"],
            ),
        ]
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(samples, metrics=["context_recall"])

        assert "context_recall" in results
        assert "context_precision" not in results


class TestRAGASMetrics:
    """Tests for RAGAS metrics list."""

    def test_ragas_metrics_list(self) -> None:
        """Should list available RAGAS metrics."""
        from embedding_tests.evaluation.rag_evaluator import RAGAS_METRICS

        assert "faithfulness" in RAGAS_METRICS
        assert "answer_relevance" in RAGAS_METRICS
        assert "context_recall" in RAGAS_METRICS
        assert "context_precision" in RAGAS_METRICS

    def test_ragas_faithfulness_not_available_without_llm(self) -> None:
        """Should skip faithfulness without LLM configured."""
        from embedding_tests.evaluation.rag_evaluator import (
            RAGEvaluator,
            RAGEvaluationSample,
        )

        samples = [
            RAGEvaluationSample(
                question="What is X?",
                contexts=["X is Y"],
                answer="X is Y",
                relevant_doc_ids=["doc1"],
                retrieved_doc_ids=["doc1"],
            )
        ]
        evaluator = RAGEvaluator()  # No LLM
        # Request faithfulness metric to verify it's skipped when no LLM
        results = evaluator.evaluate(samples, metrics=["faithfulness", "context_recall"])

        # Faithfulness should not be in results without LLM
        assert "faithfulness" not in results
        # But context_recall should still be computed
        assert "context_recall" in results
