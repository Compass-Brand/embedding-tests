"""Tests for VRAM estimation and memory management."""

from __future__ import annotations

import pytest

from embedding_tests.config.models import PrecisionLevel
from embedding_tests.hardware.memory import estimate_vram_gb, will_model_fit


class TestEstimateVram:
    """Tests for VRAM estimation."""

    def test_estimate_vram_fp16_8b_model(self) -> None:
        vram = estimate_vram_gb(params_billions=8.0, precision=PrecisionLevel.FP16)
        assert vram == 16.0

    def test_estimate_vram_int8_8b_model(self) -> None:
        vram = estimate_vram_gb(params_billions=8.0, precision=PrecisionLevel.INT8)
        assert vram == 8.0

    def test_estimate_vram_int4_8b_model(self) -> None:
        vram = estimate_vram_gb(params_billions=8.0, precision=PrecisionLevel.INT4)
        assert vram == 4.0

    def test_estimate_vram_fp16_small_model(self) -> None:
        vram = estimate_vram_gb(params_billions=0.6, precision=PrecisionLevel.FP16)
        assert vram == 1.2

    def test_estimate_vram_fp16_12b_model(self) -> None:
        vram = estimate_vram_gb(params_billions=11.76, precision=PrecisionLevel.FP16)
        assert vram == 23.52

    def test_estimate_vram_gptq_int4_8b_model(self) -> None:
        vram = estimate_vram_gb(params_billions=8.0, precision=PrecisionLevel.GPTQ_INT4)
        assert vram == 4.0

    def test_estimate_vram_awq_int4_8b_model(self) -> None:
        vram = estimate_vram_gb(params_billions=8.0, precision=PrecisionLevel.AWQ_INT4)
        assert vram == 4.0

    def test_estimate_vram_raises_for_unsupported_precision(self) -> None:
        from unittest.mock import MagicMock

        fake_precision = MagicMock()
        with pytest.raises(ValueError):
            estimate_vram_gb(params_billions=8.0, precision=fake_precision)

    def test_estimate_vram_raises_for_negative_params(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            estimate_vram_gb(params_billions=-1.0, precision=PrecisionLevel.FP16)


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

    def test_will_model_fit_raises_for_negative_params(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            will_model_fit(params_billions=-1.0, precision=PrecisionLevel.FP16, available_vram_gb=24.0)

    def test_will_model_fit_raises_for_negative_vram(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            will_model_fit(params_billions=8.0, precision=PrecisionLevel.FP16, available_vram_gb=-1.0)

    def test_will_model_fit_raises_for_negative_margin(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            will_model_fit(params_billions=8.0, precision=PrecisionLevel.FP16, available_vram_gb=24.0, safety_margin_gb=-1.0)

