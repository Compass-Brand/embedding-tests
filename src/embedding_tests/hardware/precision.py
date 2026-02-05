"""Precision strategy with hardware-aware dtype selection."""

from __future__ import annotations

from dataclasses import dataclass

from embedding_tests.config.hardware import GpuCapabilities
from embedding_tests.config.models import PrecisionLevel


@dataclass(frozen=True)
class PrecisionConfig:
    """Hardware-aware precision configuration for model loading."""

    storage_dtype: str
    compute_dtype: str
    attn_implementation: str
    use_autocast: bool
    quantization_config: dict[str, object] | None = None


def get_precision_config(
    gpu: GpuCapabilities,
    precision: PrecisionLevel,
) -> PrecisionConfig:
    """Determine optimal precision configuration for the given GPU and precision level."""
    compute_dtype = "bfloat16" if gpu.supports_bf16 else "float32"
    attn_impl = "flash_attention_2" if gpu.supports_flash_attn2 else "eager"

    if precision == PrecisionLevel.INT8:
        return PrecisionConfig(
            storage_dtype="int8",
            compute_dtype=compute_dtype,
            attn_implementation=attn_impl,
            use_autocast=False,
            quantization_config={
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
            },
        )

    if precision == PrecisionLevel.INT4:
        return PrecisionConfig(
            storage_dtype="int4",
            compute_dtype=compute_dtype,
            attn_implementation=attn_impl,
            use_autocast=False,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": compute_dtype,
                "bnb_4bit_quant_type": "nf4",
            },
        )

    if precision == PrecisionLevel.GPTQ_INT4:
        return PrecisionConfig(
            storage_dtype="gptq_int4",
            compute_dtype=compute_dtype,
            attn_implementation=attn_impl,
            use_autocast=False,
            quantization_config={"bits": 4, "backend": "gptq"},
        )

    if precision == PrecisionLevel.AWQ_INT4:
        return PrecisionConfig(
            storage_dtype="awq_int4",
            compute_dtype=compute_dtype,
            attn_implementation=attn_impl,
            use_autocast=False,
            quantization_config={"bits": 4, "backend": "awq"},
        )

    # FP16 / BF16 path
    if gpu.supports_bf16:
        return PrecisionConfig(
            storage_dtype="bfloat16",
            compute_dtype="bfloat16",
            attn_implementation="flash_attention_2" if gpu.supports_flash_attn2 else "eager",
            use_autocast=True,
            quantization_config=None,
        )

    # P40 and similar: FP16 storage, FP32 compute, eager attention
    return PrecisionConfig(
        storage_dtype="float16",
        compute_dtype="float32",
        attn_implementation="eager",
        use_autocast=False,
        quantization_config=None,
    )
