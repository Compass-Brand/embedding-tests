"""GPU hardware detection and capability checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# BF16 requires compute capability >= 8.0 (Ampere+)
_BF16_MIN_CC = (8, 0)
# Flash Attention 2 requires compute capability >= 8.0 (Ampere+)
_FLASH_ATTN2_MIN_CC = (8, 0)


@dataclass(frozen=True)
class GpuCapabilities:
    """Detected GPU capabilities."""

    device_name: str
    compute_capability: tuple[int, int]
    total_vram_gb: float
    supports_bf16: bool
    supports_flash_attn2: bool


def detect_gpu(device_index: int = 0) -> GpuCapabilities | None:
    """Detect GPU capabilities. Returns None if no CUDA device available."""
    if not torch.cuda.is_available():
        return None

    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        raise ValueError(
            f"device_index {device_index} out of range; only {device_count} GPU(s) available"
        )

    device_name = torch.cuda.get_device_name(device_index)
    cc = torch.cuda.get_device_capability(device_index)
    props = torch.cuda.get_device_properties(device_index)
    total_vram_gb = props.total_memory / (1024**3)

    supports_bf16 = cc >= _BF16_MIN_CC
    supports_flash_attn2 = cc >= _FLASH_ATTN2_MIN_CC

    return GpuCapabilities(
        device_name=device_name,
        compute_capability=cc,
        total_vram_gb=total_vram_gb,
        supports_bf16=supports_bf16,
        supports_flash_attn2=supports_flash_attn2,
    )
