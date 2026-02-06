"""Experiment orchestrator with GPU memory management."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

from embedding_tests.config.hardware import GpuCapabilities, detect_gpu
from embedding_tests.config.models import ModelConfig, ModelType, PrecisionLevel
from embedding_tests.evaluation.metrics import (
    compute_aggregate_stats,
    f1_at_k,
    mean_average_precision,
    mean_success_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
    success_at_k,
)
from embedding_tests.hardware.precision import get_precision_config
from embedding_tests.models.loader import load_model
from embedding_tests.pipeline.rag import RagPipeline
from embedding_tests.runner.checkpoint import clear_checkpoints, is_completed, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates experiments across model/precision combinations."""

    def __init__(
        self,
        model_configs: list[ModelConfig],
        precisions: list[PrecisionLevel],
        corpus: list[dict[str, Any]],
        queries: list[dict[str, Any]],
        *,
        checkpoint_dir: Path | None = None,
        top_k: int = 10,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        clear_on_success: bool = False,
    ) -> None:
        self._models = model_configs
        self._precisions = precisions
        self._corpus = corpus
        self._queries = queries
        self._checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._clear_on_success = clear_on_success

    def run(self) -> list[dict[str, Any]]:
        """Run all model/precision combinations."""
        gpu = detect_gpu()
        results: list[dict[str, Any]] = []

        for model_config in self._models:
            for precision in self._precisions:
                # Skip if precision not supported by this model
                if precision not in model_config.supported_precisions:
                    logger.warning(
                        "Skipping %s at %s (not in supported precisions)",
                        model_config.name,
                        precision.value,
                    )
                    continue

                result = self._run_single(model_config, precision, gpu)
                results.append(result)

        # Clear checkpoints if all runs completed successfully
        if self._clear_on_success and results:
            all_completed = all(r.get("status") == "completed" for r in results)
            if all_completed:
                cleared = clear_checkpoints(self._checkpoint_dir)
                logger.info("Cleared %d checkpoints after successful run", cleared)

        return results

    def _run_single(
        self,
        model_config: ModelConfig,
        precision: PrecisionLevel,
        gpu: GpuCapabilities | None,
    ) -> dict[str, Any]:
        """Run a single model/precision combination."""
        name = model_config.name
        prec = precision.value

        # Check checkpoint
        if is_completed(self._checkpoint_dir, name, prec):
            logger.info("Skipping %s/%s (checkpointed)", name, prec)
            checkpoint = load_checkpoint(self._checkpoint_dir, name, prec)
            if checkpoint:
                return {
                    "model": checkpoint.get("model_name", name),
                    "precision": checkpoint.get("precision", prec),
                    "status": checkpoint.get("status"),
                    "results": checkpoint.get("results"),
                    "mrr": checkpoint.get("mrr", 0.0),
                    "total_time": checkpoint.get("total_time", 0.0),
                }
            logger.warning("Checkpoint marked complete but data missing for %s/%s", name, prec)
            return {"model": name, "precision": prec, "status": "completed_missing_data", "results": None}

        logger.info("Running %s at %s precision", name, prec)

        # Check if model is a reranker (no encode method) before loading
        if model_config.model_type == ModelType.MULTIMODAL_RERANKER:
            logger.warning(
                "Skipping %s/%s (reranker models not supported in RAG pipeline)",
                name,
                prec,
            )
            return {
                "model": name,
                "precision": prec,
                "error": "Reranker models not supported in RAG pipeline",
            }

        model = None
        try:
            # GPU is required for model loading; skip combination if unavailable
            if gpu is None:
                return {"model": name, "precision": prec, "error": "No GPU detected"}
            precision_config = get_precision_config(gpu, precision)

            model = load_model(model_config, precision_config)

            pipeline = RagPipeline(
                embedding_model=model,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                top_k=self._top_k,
            )
            rag_result = pipeline.run(self._corpus, self._queries)

            # Define k values for multi-k metrics
            k_values = [1, 3, 5, 10, 20]

            # Compute metrics per query
            per_query_metrics: dict[str, Any] = {}
            mrr_inputs: list[tuple[list[str], set[str]]] = []

            # Collectors for aggregate statistics
            recall_scores: dict[int, list[float]] = {k: [] for k in k_values}
            precision_scores: dict[int, list[float]] = {k: [] for k in k_values}
            ndcg_scores: dict[int, list[float]] = {k: [] for k in k_values}
            f1_scores: dict[int, list[float]] = {k: [] for k in k_values}
            r_precision_scores: list[float] = []

            for qr in rag_result.query_results:
                relevant = set(qr.relevant_doc_ids)
                relevance_scores = {doc_id: 1.0 for doc_id in qr.relevant_doc_ids}

                query_metrics: dict[str, Any] = {}

                # Compute metrics at multiple k values
                for k in k_values:
                    r_k = recall_at_k(qr.retrieved_doc_ids, relevant, k=k)
                    p_k = precision_at_k(qr.retrieved_doc_ids, relevant, k=k)
                    ndcg = ndcg_at_k(qr.retrieved_doc_ids, relevance_scores, k=k)
                    f1 = f1_at_k(qr.retrieved_doc_ids, relevant, k=k)
                    s_k = success_at_k(qr.retrieved_doc_ids, relevant, k=k)

                    query_metrics[f"recall_at_{k}"] = r_k
                    query_metrics[f"precision_at_{k}"] = p_k
                    query_metrics[f"ndcg_at_{k}"] = ndcg
                    query_metrics[f"f1_at_{k}"] = f1
                    query_metrics[f"success_at_{k}"] = s_k

                    # Collect for aggregation
                    recall_scores[k].append(r_k)
                    precision_scores[k].append(p_k)
                    ndcg_scores[k].append(ndcg)
                    f1_scores[k].append(f1)

                # R-Precision (single value per query)
                r_prec = r_precision(qr.retrieved_doc_ids, relevant)
                query_metrics["r_precision"] = r_prec
                r_precision_scores.append(r_prec)

                per_query_metrics[qr.query_id] = query_metrics
                mrr_inputs.append((qr.retrieved_doc_ids, relevant))

            # Compute aggregate metrics across all queries
            mrr_score = mrr(mrr_inputs)
            map_score = mean_average_precision(mrr_inputs)

            # Build aggregate statistics
            aggregate: dict[str, Any] = {
                "mrr": mrr_score,
                "map": map_score,
                "r_precision": compute_aggregate_stats(r_precision_scores),
            }

            # Add per-k aggregate statistics
            for k in k_values:
                aggregate[f"recall_at_{k}"] = compute_aggregate_stats(recall_scores[k])
                aggregate[f"precision_at_{k}"] = compute_aggregate_stats(precision_scores[k])
                aggregate[f"ndcg_at_{k}"] = compute_aggregate_stats(ndcg_scores[k])
                aggregate[f"f1_at_{k}"] = compute_aggregate_stats(f1_scores[k])
                aggregate[f"success_at_{k}"] = mean_success_at_k(mrr_inputs, k=k)

            # Performance metrics
            performance = {
                "total_time_seconds": rag_result.total_time_seconds,
                "embedding_time_seconds": rag_result.embedding_time_seconds,
                "num_corpus_chunks": rag_result.num_corpus_chunks,
                "num_queries": len(rag_result.query_results),
                "queries_per_second": len(rag_result.query_results) / rag_result.total_time_seconds
                if rag_result.total_time_seconds > 0
                else 0.0,
            }

            result = {
                "model": name,
                "precision": prec,
                "status": "completed",
                "results": per_query_metrics,
                "aggregate": aggregate,
                "performance": performance,
                # Keep top-level for backward compatibility
                "mrr": mrr_score,
                "map": map_score,
                "total_time": rag_result.total_time_seconds,
            }

            save_checkpoint(
                self._checkpoint_dir, name, prec, "completed", per_query_metrics,
                mrr=mrr_score, total_time=rag_result.total_time_seconds
            )
            return result

        except Exception as e:
            logger.error("Failed %s/%s: %s", name, prec, e, exc_info=True)
            save_checkpoint(self._checkpoint_dir, name, prec, "failed", {"error": str(e)})
            return {"model": name, "precision": prec, "error": str(e)}

        finally:
            if model is not None:
                model.unload()
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
