"""Tests for VRAM estimation and memory management."""

from __future__ import annotations

import pytest

from embedding_tests.config.models import PrecisionLevel
from embedding_tests.hardware.memory import estimate_vram_gb, will_model_fit


class TestEstimateVram:
    """Tests for VRAM estimation."""

    def test_estimate_vram_fp16_8b_model(self) -> None:
        vram = estimate_vram_gb(params_billions=8.0, precision=PrecisionLevel.FP16)
        assert vram == pytest.approx(16.0, rel=0.1)

    def test_estimate_vram_int8_8b_model(self) -> None:
        vram = estimate_vram_gb(params_billions=8.0, precision=PrecisionLevel.INT8)
        assert vram == pytest.approx(8.0, rel=0.1)

    def test_estimate_vram_int4_8b_model(self) -> None:
        vram = estimate_vram_gb(params_billions=8.0, precision=PrecisionLevel.INT4)
        assert vram == pytest.approx(4.0, rel=0.1)

    def test_estimate_vram_fp16_small_model(self) -> None:
        vram = estimate_vram_gb(params_billions=0.6, precision=PrecisionLevel.FP16)
        assert vram == pytest.approx(1.2, rel=0.1)

    def test_estimate_vram_fp16_12b_model(self) -> None:
        vram = estimate_vram_gb(params_billions=11.76, precision=PrecisionLevel.FP16)
        assert vram == pytest.approx(23.52, rel=0.1)


class TestWillModelFit:
    """Tests for model fit checking."""

    def test_will_model_fit_true_for_small_model(self) -> None:
        fits = will_model_fit(
            params_billions=0.6,
            precision=PrecisionLevel.FP16,
            available_vram_gb=24.0,
        )
        assert fits is True

    def test_will_model_fit_false_for_kalm_fp16(self) -> None:
        fits = will_model_fit(
            params_billions=11.76,
            precision=PrecisionLevel.FP16,
            available_vram_gb=24.0,
        )
        assert fits is False

    def test_will_model_fit_with_safety_margin(self) -> None:
        # 8B FP16 = ~16GB, with 2GB margin needs 18GB. In 20GB this fits.
        fits = will_model_fit(
            params_billions=8.0,
            precision=PrecisionLevel.FP16,
            available_vram_gb=20.0,
            safety_margin_gb=2.0,
        )
        assert fits is True

    def test_will_model_fit_fails_with_safety_margin(self) -> None:
        # 8B FP16 = ~16GB, with 2GB margin needs 18GB. In 17GB this doesn't fit.
        fits = will_model_fit(
            params_billions=8.0,
            precision=PrecisionLevel.FP16,
            available_vram_gb=17.0,
            safety_margin_gb=2.0,
        )
        assert fits is False
