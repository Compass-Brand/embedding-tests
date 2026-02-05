"""VL embedding wrapper for Qwen3-VL multimodal embedding models."""

from __future__ import annotations

import gc
import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from embedding_tests.config.models import ModelConfig
from embedding_tests.hardware.precision import PrecisionConfig

logger = logging.getLogger(__name__)


class VLEmbeddingWrapper:
    """Wrapper for Qwen3-VL embedding models using transformers directly."""

    def __init__(self, config: ModelConfig, precision: PrecisionConfig) -> None:
        self._config = config
        self._precision = precision
        self._embedding_dim = config.embedding_dim

        _QUANTIZED_DTYPES = {"int4", "int8", "gptq_int4", "awq_int4"}

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

        self._model = AutoModel.from_pretrained(config.model_id, **load_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=config.trust_remote_code
        )

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Encode texts into embedding vectors."""
        all_embeddings: list[np.ndarray] = []

        if is_query and self._config.query_instruction:
            texts = [f"{self._config.query_instruction}{t}" for t in texts]

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            device = self._model.get_input_embeddings().weight.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Use last hidden state, mean pooling over sequence
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            # Normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim

    def unload(self) -> None:
        """Release GPU memory."""
        del self._model
        del self._tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
