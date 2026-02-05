"""Tests for model configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from embedding_tests.config.models import (
    ModelConfig,
    ModelType,
    PrecisionLevel,
    load_all_model_configs,
    load_model_config,
)


class TestModelType:
    """Tests for ModelType enum."""

    def test_model_type_has_text_embedding(self) -> None:
        assert ModelType.TEXT_EMBEDDING.value == "text_embedding"

    def test_model_type_has_multimodal_embedding(self) -> None:
        assert ModelType.MULTIMODAL_EMBEDDING.value == "multimodal_embedding"

    def test_model_type_has_multimodal_reranker(self) -> None:
        assert ModelType.MULTIMODAL_RERANKER.value == "multimodal_reranker"


class TestPrecisionLevel:
    """Tests for PrecisionLevel enum."""

    def test_precision_has_fp16(self) -> None:
        assert PrecisionLevel.FP16.value == "fp16"

    def test_precision_has_int8(self) -> None:
        assert PrecisionLevel.INT8.value == "int8"

    def test_precision_has_int4(self) -> None:
        assert PrecisionLevel.INT4.value == "int4"

    def test_precision_has_gptq_int4(self) -> None:
        assert PrecisionLevel.GPTQ_INT4.value == "gptq_int4"

    def test_precision_has_awq_int4(self) -> None:
        assert PrecisionLevel.AWQ_INT4.value == "awq_int4"


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_creation_with_all_fields(self) -> None:
        config = ModelConfig(
            name="test-model",
            model_id="org/test-model",
            model_type=ModelType.TEXT_EMBEDDING,
            params_billions=0.6,
            embedding_dim=1024,
            supported_precisions=[PrecisionLevel.FP16, PrecisionLevel.INT8],
            trust_remote_code=True,
            query_instruction="Represent this query: ",
        )
        assert config.name == "test-model"
        assert config.model_id == "org/test-model"
        assert config.model_type == ModelType.TEXT_EMBEDDING
        assert config.params_billions == 0.6
        assert config.embedding_dim == 1024
        assert config.supported_precisions == [PrecisionLevel.FP16, PrecisionLevel.INT8]
        assert config.trust_remote_code is True
        assert config.query_instruction == "Represent this query: "

    def test_model_config_defaults(self) -> None:
        config = ModelConfig(
            name="test",
            model_id="org/test",
            model_type=ModelType.TEXT_EMBEDDING,
            params_billions=1.0,
            embedding_dim=768,
            supported_precisions=[PrecisionLevel.FP16],
        )
        assert config.trust_remote_code is True
        assert config.query_instruction is None
        assert config.max_seq_length is None

    def test_model_config_validation_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelConfig(
                name="",
                model_id="org/test",
                model_type=ModelType.TEXT_EMBEDDING,
                params_billions=1.0,
                embedding_dim=768,
                supported_precisions=[PrecisionLevel.FP16],
            )

    def test_model_config_validation_rejects_negative_params(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            ModelConfig(
                name="test",
                model_id="org/test",
                model_type=ModelType.TEXT_EMBEDDING,
                params_billions=-1.0,
                embedding_dim=768,
                supported_precisions=[PrecisionLevel.FP16],
            )


class TestLoadModelConfig:
    """Tests for YAML config loading."""

    def test_load_model_config_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
name: test-model
model_id: org/test-model
model_type: text_embedding
params_billions: 0.6
embedding_dim: 1024
supported_precisions:
  - fp16
  - int8
trust_remote_code: true
query_instruction: "query: "
"""
        config_file = tmp_path / "test_model.yaml"
        config_file.write_text(yaml_content)
        config = load_model_config(config_file)
        assert config.name == "test-model"
        assert config.model_type == ModelType.TEXT_EMBEDDING
        assert config.params_billions == 0.6
        assert PrecisionLevel.FP16 in config.supported_precisions
        assert PrecisionLevel.INT8 in config.supported_precisions

    def test_load_all_model_configs(self, configs_dir: Path) -> None:
        configs = load_all_model_configs(configs_dir / "models")
        assert len(configs) >= 2  # At least some configs exist
        names = {c.name for c in configs}
        # Verify configs have names and required fields
        for c in configs:
            assert c.name
            assert c.model_id
            assert c.embedding_dim > 0 or c.model_type == ModelType.MULTIMODAL_RERANKER

    def test_load_model_config_validates_unknown_precision(self, tmp_path: Path) -> None:
        yaml_content = """
name: bad-model
model_id: org/bad
model_type: text_embedding
params_billions: 1.0
embedding_dim: 768
supported_precisions:
  - fp999
"""
        config_file = tmp_path / "bad_model.yaml"
        config_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="precision"):
            load_model_config(config_file)
