"""Sentence-transformers wrapper for text embedding models."""

from __future__ import annotations

import gc
import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from embedding_tests.config.models import ModelConfig
from embedding_tests.hardware.precision import PrecisionConfig

logger = logging.getLogger(__name__)

_QUANTIZED_DTYPES = {"int4", "int8", "gptq_int4", "awq_int4"}


class SentenceTransformerWrapper:
    """Wrapper around sentence-transformers for hardware-aware loading."""

    def __init__(self, config: ModelConfig, precision: PrecisionConfig) -> None:
        self._config = config
        self._precision = precision
        self._embedding_dim = config.embedding_dim

        if precision.storage_dtype in _QUANTIZED_DTYPES:
            torch_dtype = torch.float16
        elif hasattr(torch, precision.storage_dtype):
            torch_dtype = getattr(torch, precision.storage_dtype)
        else:
            raise ValueError(f"Invalid storage dtype: {precision.storage_dtype}")

        model_kwargs: dict[str, object] = {
            "torch_dtype": torch_dtype,
            "attn_implementation": precision.attn_implementation,
        }

        if precision.quantization_config is not None:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                **precision.quantization_config
            )

        tokenizer_kwargs: dict[str, object] = {}
        if config.padding_side:
            tokenizer_kwargs["padding_side"] = config.padding_side

        self._model = SentenceTransformer(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode texts into embedding vectors."""
        prompt = self._config.query_instruction if is_query else None

        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt=prompt,
            show_progress_bar=False,
        )

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim

    def unload(self) -> None:
        """Release GPU memory."""
        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
