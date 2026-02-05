"""VL reranker wrapper for Qwen3-VL reranker models.

Uses Qwen3VLForConditionalGeneration with a binary classification head
extracted from the LM head's "yes"/"no" token weights, following the
official Qwen3-VL-Reranker scoring pattern.
"""

from __future__ import annotations

import gc
import logging

import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from embedding_tests.config.models import ModelConfig
from embedding_tests.hardware.precision import PrecisionConfig

logger = logging.getLogger(__name__)

_QUANTIZED_DTYPES = {"int4", "int8", "gptq_int4", "awq_int4"}

_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)

_DEFAULT_INSTRUCTION = (
    "Given a search query, retrieve relevant candidates that answer the query."
)


class VLRerankerWrapper:
    """Wrapper for Qwen3-VL reranker models.

    Loads ``Qwen3VLForConditionalGeneration``, extracts the base transformer
    and builds a lightweight binary scorer from the LM head's *yes*/*no*
    token weights.  Text-only scoring is done per document via the chat
    template used by the official reranker scripts.
    """

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
            "device_map": "auto",
        }

        if precision.quantization_config is not None:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                **precision.quantization_config
            )

        # Load the full conditional-generation model
        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_id, **load_kwargs
        )

        # Keep only the base transformer (no LM head needed at inference)
        self._model = lm.model
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
            padding_side="left",
        )

        # Build a 1-D linear scorer: score = (yes − no) · hidden_state
        vocab = self._tokenizer.get_vocab()
        token_yes = vocab["yes"]
        token_no = vocab["no"]
        lm_weights = lm.lm_head.weight.data
        dim = lm_weights.size(1)

        self._score_linear = torch.nn.Linear(dim, 1, bias=False)
        with torch.no_grad():
            self._score_linear.weight[0] = lm_weights[token_yes] - lm_weights[token_no]
        self._score_linear.eval()

        self._device = next(self._model.parameters()).device
        self._score_linear = self._score_linear.to(
            device=self._device, dtype=self._model.dtype
        )

        # Free the full model (LM head, tied embeddings, etc.)
        del lm

    # ------------------------------------------------------------------
    # Public API (satisfies RerankerModel protocol)
    # ------------------------------------------------------------------

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

        instruction = self._config.query_instruction or _DEFAULT_INSTRUCTION
        max_length = self._config.max_seq_length or 8192

        scores: list[float] = []
        for doc in documents:
            messages = self._build_messages(instruction, query, doc)
            score = self._score_single(messages, max_length)
            scores.append(score)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]

    def unload(self) -> None:
        """Release GPU memory."""
        self._model = None
        self._score_linear = None
        self._tokenizer = None
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_single(self, messages: list[dict], max_length: int) -> float:
        """Score a single query-document pair."""
        text = self._tokenizer.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        last_hidden = outputs.last_hidden_state[:, -1]
        score = self._score_linear(last_hidden)
        return torch.sigmoid(score).squeeze().cpu().item()

    @staticmethod
    def _build_messages(
        instruction: str, query: str, document: str
    ) -> list[dict]:
        """Build chat messages matching the official reranker format."""
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<Instruct>: {instruction}"},
                    {"type": "text", "text": "<Query>:"},
                    {"type": "text", "text": query},
                    {"type": "text", "text": "\n<Document>:"},
                    {"type": "text", "text": document},
                ],
            },
        ]
