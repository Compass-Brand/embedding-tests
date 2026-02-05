"""VRAM estimation and memory management."""

from __future__ import annotations

from embedding_tests.config.models import PrecisionLevel

# Bytes per parameter for each precision level
_BYTES_PER_PARAM: dict[PrecisionLevel, float] = {
    PrecisionLevel.FP16: 2.0,
    PrecisionLevel.INT8: 1.0,
    PrecisionLevel.INT4: 0.5,
    PrecisionLevel.GPTQ_INT4: 0.5,
    PrecisionLevel.AWQ_INT4: 0.5,
}


def estimate_vram_gb(params_billions: float, precision: PrecisionLevel) -> float:
    """Estimate VRAM usage in GB for a model at a given precision.

    Uses a simple formula: params * bytes_per_param.
    Does not account for activation memory or KV cache overhead.
    """
    bytes_per_param = _BYTES_PER_PARAM.get(precision)
    if bytes_per_param is None:
        raise ValueError(
            f"Unsupported precision level: {precision}. "
            f"Supported: {list(_BYTES_PER_PARAM.keys())}"
        )
    return params_billions * bytes_per_param


def will_model_fit(
    params_billions: float,
    precision: PrecisionLevel,
    available_vram_gb: float,
    safety_margin_gb: float = 2.0,
) -> bool:
    """Check if a model will fit in available VRAM with safety margin.

    The default ``safety_margin_gb`` of 2.0 GB accounts for CUDA context
    overhead and activation memory. Increase this for models with large
    intermediate activations or when running alongside other GPU workloads.
    """
    if params_billions < 0:
        raise ValueError(f"params_billions must be non-negative, got {params_billions}")
    if available_vram_gb < 0:
        raise ValueError(f"available_vram_gb must be non-negative, got {available_vram_gb}")
    if safety_margin_gb < 0:
        raise ValueError(f"safety_margin_gb must be non-negative, got {safety_margin_gb}")
    estimated = estimate_vram_gb(params_billions, precision)
    return (estimated + safety_margin_gb) <= available_vram_gb
