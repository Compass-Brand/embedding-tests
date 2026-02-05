"""VL reranker wrapper for Qwen3-VL reranker models."""

from __future__ import annotations

import gc
import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from embedding_tests.config.models import ModelConfig
from embedding_tests.hardware.precision import PrecisionConfig

logger = logging.getLogger(__name__)

_QUANTIZED_DTYPES = {"int4", "int8", "gptq_int4", "awq_int4"}


class VLRerankerWrapper:
    """Wrapper for Qwen3-VL reranker models using transformers directly."""

    def __init__(self, config: ModelConfig, precision: PrecisionConfig) -> None:
        self._config = config
        self._precision = precision

        if precision.storage_dtype in _QUANTIZED_DTYPES:
            dtype = torch.float16
        elif hasattr(torch, precision.storage_dtype):
            dtype = getattr(torch, precision.storage_dtype)
        else:
            raise ValueError(f"Invalid storage dtype: {precision.storage_dtype}")

        load_kwargs: dict[str, object] = {
            "torch_dtype": dtype,
            "attn_implementation": precision.attn_implementation,
            "trust_remote_code": config.trust_remote_code,
        }

        load_kwargs["device_map"] = "auto"

        if precision.quantization_config is not None:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                **precision.quantization_config
            )

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
        if not documents:
            return []

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        top_k = min(top_k, len(documents))

        inputs = self._tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            max_length=self._config.max_seq_length,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits.squeeze(-1)
        if logits.dim() == 0:
            scores = [logits.item()]
        else:
            scores = logits.tolist()

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]

    def unload(self) -> None:
        """Release GPU memory."""
        del self._model
        del self._tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
