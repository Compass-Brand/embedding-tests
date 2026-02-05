"""Experiment orchestrator with GPU memory management."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

from embedding_tests.config.hardware import GpuCapabilities, detect_gpu
from embedding_tests.config.models import ModelConfig, ModelType, PrecisionLevel
from embedding_tests.evaluation.metrics import recall_at_k, precision_at_k
from embedding_tests.hardware.precision import get_precision_config
from embedding_tests.models.loader import load_model
from embedding_tests.pipeline.rag import RagPipeline
from embedding_tests.runner.checkpoint import is_completed, load_checkpoint, save_checkpoint

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
    ) -> None:
        self._models = model_configs
        self._precisions = precisions
        self._corpus = corpus
        self._queries = queries
        self._checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

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
                }
            return {"model": name, "precision": prec, "skipped": True}

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

            # Compute metrics
            metrics: dict[str, Any] = {}
            for qr in rag_result.query_results:
                relevant = set(qr.relevant_doc_ids)
                r_k = recall_at_k(qr.retrieved_doc_ids, relevant, k=self._top_k)
                p_k = precision_at_k(qr.retrieved_doc_ids, relevant, k=self._top_k)
                metrics[qr.query_id] = {
                    f"recall_at_{self._top_k}": r_k,
                    f"precision_at_{self._top_k}": p_k,
                }

            result = {
                "model": name,
                "precision": prec,
                "status": "completed",
                "results": metrics,
                "total_time": rag_result.total_time_seconds,
            }

            save_checkpoint(self._checkpoint_dir, name, prec, "completed", metrics)
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
