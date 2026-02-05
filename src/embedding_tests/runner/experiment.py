"""Experiment orchestrator with GPU memory management."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

from embedding_tests.config.hardware import detect_gpu
from embedding_tests.config.models import ModelConfig, PrecisionLevel
from embedding_tests.evaluation.metrics import recall_at_k, mrr, precision_at_k
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
                result = self._run_single(model_config, precision, gpu)
                results.append(result)

        return results

    def _run_single(
        self,
        model_config: ModelConfig,
        precision: PrecisionLevel,
        gpu: Any,
    ) -> dict[str, Any]:
        """Run a single model/precision combination."""
        name = model_config.name
        prec = precision.value

        # Check checkpoint
        if is_completed(self._checkpoint_dir, name, prec):
            logger.info("Skipping %s/%s (checkpointed)", name, prec)
            checkpoint = load_checkpoint(self._checkpoint_dir, name, prec)
            return checkpoint if checkpoint else {"model": name, "precision": prec, "skipped": True}

        logger.info("Running %s at %s precision", name, prec)

        try:
            precision_config = get_precision_config(gpu, precision) if gpu else None
            if precision_config is None:
                return {"model": name, "precision": prec, "error": "No GPU detected"}

            model = load_model(model_config, precision_config)

            pipeline = RagPipeline(
                embedding_model=model,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                top_k=self._top_k,
            )
            rag_result = pipeline.run(self._corpus, self._queries)

            # Compute metrics
            metrics: dict[str, float] = {}
            for qr in rag_result.query_results:
                relevant = set(qr.relevant_doc_ids)
                r10 = recall_at_k(qr.retrieved_doc_ids, relevant, k=10)
                p10 = precision_at_k(qr.retrieved_doc_ids, relevant, k=10)
                metrics[qr.query_id] = r10

            result = {
                "model": name,
                "precision": prec,
                "status": "completed",
                "results": metrics,
                "total_time": rag_result.total_time_seconds,
            }

            save_checkpoint(self._checkpoint_dir, name, prec, "completed", metrics)

            # Cleanup
            model.unload()
            gc.collect()

            return result

        except Exception as e:
            logger.error("Failed %s/%s: %s", name, prec, e)
            error_result = {"model": name, "precision": prec, "error": str(e)}
            return error_result
