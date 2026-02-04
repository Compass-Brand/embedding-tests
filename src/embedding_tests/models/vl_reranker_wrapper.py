"""VL reranker wrapper for Qwen3-VL reranker models."""

from __future__ import annotations

import gc
import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from embedding_tests.config.models import ModelConfig
from embedding_tests.hardware.precision import PrecisionConfig

logger = logging.getLogger(__name__)


class VLRerankerWrapper:
    """Wrapper for Qwen3-VL reranker models using transformers directly."""

    def __init__(self, config: ModelConfig, precision: PrecisionConfig) -> None:
        self._config = config
        self._precision = precision

        dtype = getattr(torch, precision.storage_dtype, torch.float16)

        load_kwargs: dict[str, object] = {
            "torch_dtype": dtype,
            "attn_implementation": precision.attn_implementation,
            "trust_remote_code": config.trust_remote_code,
        }

        if precision.quantization_config is not None:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                **precision.quantization_config
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id, **load_kwargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=config.trust_remote_code
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query."""
        pairs = [[query, doc] for doc in documents]
        scores: list[float] = []

        for pair in pairs:
            inputs = self._tokenizer(
                pair[0],
                pair[1],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            score = outputs.logits.squeeze(-1).item()
            scores.append(score)

        # Create (index, score) tuples sorted by score descending
        indexed_scores = [(i, s) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]

    def unload(self) -> None:
        """Release GPU memory."""
        del self._model
        del self._tokenizer
        gc.collect()
        torch.cuda.empty_cache()
