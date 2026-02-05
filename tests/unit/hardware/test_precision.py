"""Tests for precision strategy with hardware-aware dtype selection."""

from __future__ import annotations

import pytest

from embedding_tests.config.hardware import GpuCapabilities
from embedding_tests.config.models import PrecisionLevel
from embedding_tests.hardware.precision import PrecisionConfig, get_precision_config


@pytest.fixture
def p40_caps() -> GpuCapabilities:
    return GpuCapabilities(
        device_name="Tesla P40",
        compute_capability=(6, 1),
        total_vram_gb=24.0,
        supports_bf16=False,
        supports_flash_attn2=False,
    )


@pytest.fixture
def a100_caps() -> GpuCapabilities:
    return GpuCapabilities(
        device_name="NVIDIA A100",
        compute_capability=(8, 0),
        total_vram_gb=80.0,
        supports_bf16=True,
        supports_flash_attn2=True,
    )


class TestPrecisionConfigP40:
    """Tests for P40-specific precision strategy."""

    def test_p40_precision_uses_fp16_storage(self, p40_caps: GpuCapabilities) -> None:
        config = get_precision_config(p40_caps, PrecisionLevel.FP16)
        assert config.storage_dtype == "float16"

    def test_p40_precision_uses_fp32_compute(self, p40_caps: GpuCapabilities) -> None:
        config = get_precision_config(p40_caps, PrecisionLevel.FP16)
        assert config.compute_dtype == "float32"

    def test_p40_precision_uses_eager_attention(self, p40_caps: GpuCapabilities) -> None:
        config = get_precision_config(p40_caps, PrecisionLevel.FP16)
        assert config.attn_implementation == "eager"

    def test_p40_no_autocast(self, p40_caps: GpuCapabilities) -> None:
        config = get_precision_config(p40_caps, PrecisionLevel.FP16)
        assert config.use_autocast is False

    def test_int8_quantization_config(self, p40_caps: GpuCapabilities) -> None:
        config = get_precision_config(p40_caps, PrecisionLevel.INT8)
        assert config.quantization_config is not None
        assert config.quantization_config["load_in_8bit"] is True

    def test_int4_quantization_config(self, p40_caps: GpuCapabilities) -> None:
        config = get_precision_config(p40_caps, PrecisionLevel.INT4)
        assert config.quantization_config is not None
        assert config.quantization_config["load_in_4bit"] is True
        assert config.quantization_config["bnb_4bit_compute_dtype"] == "float32"


class TestPrecisionConfigAmpere:
    """Tests for Ampere+ GPU precision strategy."""

    def test_ampere_can_use_bf16_and_flash(self, a100_caps: GpuCapabilities) -> None:
        config = get_precision_config(a100_caps, PrecisionLevel.FP16)
        # Ampere can use bf16 storage and flash attention
        assert config.storage_dtype == "bfloat16"
        assert config.attn_implementation == "flash_attention_2"
        assert config.use_autocast is True


class TestPrecisionConfigFields:
    """Tests for PrecisionConfig dataclass."""

    def test_precision_config_has_required_fields(self, p40_caps: GpuCapabilities) -> None:
        config = get_precision_config(p40_caps, PrecisionLevel.FP16)
        assert isinstance(config.storage_dtype, str)
        assert isinstance(config.compute_dtype, str)
        assert isinstance(config.attn_implementation, str)
        assert isinstance(config.use_autocast, bool)
        assert config.quantization_config is None or isinstance(config.quantization_config, dict)
