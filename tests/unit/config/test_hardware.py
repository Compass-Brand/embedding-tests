"""Tests for GPU hardware detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from embedding_tests.config.hardware import GpuCapabilities, detect_gpu


class TestGpuCapabilities:
    """Tests for GpuCapabilities dataclass."""

    def test_gpu_capabilities_fields(self) -> None:
        caps = GpuCapabilities(
            device_name="Tesla P40",
            compute_capability=(6, 1),
            total_vram_gb=24.0,
            supports_bf16=False,
            supports_flash_attn2=False,
        )
        assert caps.device_name == "Tesla P40"
        assert caps.compute_capability == (6, 1)
        assert caps.total_vram_gb == 24.0
        assert caps.supports_bf16 is False
        assert caps.supports_flash_attn2 is False


class TestDetectGpu:
    """Tests for GPU detection function."""

    @patch("embedding_tests.config.hardware.torch")
    def test_gpu_capabilities_detection_with_cuda(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Tesla P40"
        mock_torch.cuda.get_device_capability.return_value = (6, 1)
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024**3  # 24 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props

        caps = detect_gpu()
        assert caps is not None
        assert caps.device_name == "Tesla P40"
        assert caps.compute_capability == (6, 1)
        assert caps.total_vram_gb == pytest.approx(24.0, rel=0.01)

    @patch("embedding_tests.config.hardware.torch")
    def test_gpu_capabilities_no_cuda(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = False
        caps = detect_gpu()
        assert caps is None

    @patch("embedding_tests.config.hardware.torch")
    def test_p40_does_not_support_bf16(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Tesla P40"
        mock_torch.cuda.get_device_capability.return_value = (6, 1)
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = mock_props

        caps = detect_gpu()
        assert caps is not None
        assert caps.supports_bf16 is False

    @patch("embedding_tests.config.hardware.torch")
    def test_p40_does_not_support_flash_attn2(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Tesla P40"
        mock_torch.cuda.get_device_capability.return_value = (6, 1)
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = mock_props

        caps = detect_gpu()
        assert caps is not None
        assert caps.supports_flash_attn2 is False

    @patch("embedding_tests.config.hardware.torch")
    def test_ampere_supports_bf16_and_flash(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        mock_props = MagicMock()
        mock_props.total_memory = 80 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = mock_props

        caps = detect_gpu()
        assert caps is not None
        assert caps.supports_bf16 is True
        assert caps.supports_flash_attn2 is True
