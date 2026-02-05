"""Tests for experiment configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from embedding_tests.config.experiment import ExperimentConfig, load_experiment_config


class TestLoadExperimentConfig:
    """Tests for experiment config loading."""

    def test_load_experiment_config_from_yaml(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "test_model.yaml").write_text("""
name: test-model
model_id: org/test
model_type: text_embedding
params_billions: 0.6
embedding_dim: 1024
supported_precisions:
  - fp16
""")
        experiment_yaml = tmp_path / "experiment.yaml"
        experiment_yaml.write_text("""
name: test-experiment
description: A test experiment
models:
  - test-model
precisions:
  - fp16
pipeline:
  chunk_size: 512
  chunk_overlap: 50
  retrieval_top_k: 10
""")
        config = load_experiment_config(experiment_yaml, models_dir)
        assert config.name == "test-experiment"
        assert config.description == "A test experiment"

    def test_experiment_config_resolves_model_references(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "m1.yaml").write_text("""
name: model-a
model_id: org/a
model_type: text_embedding
params_billions: 1.0
embedding_dim: 768
supported_precisions:
  - fp16
""")
        experiment_yaml = tmp_path / "exp.yaml"
        experiment_yaml.write_text("""
name: exp
description: test
models:
  - model-a
precisions:
  - fp16
""")
        config = load_experiment_config(experiment_yaml, models_dir)
        assert len(config.models) == 1
        assert config.models[0].name == "model-a"
        assert config.models[0].model_id == "org/a"

    def test_experiment_config_validates_precisions(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "m.yaml").write_text("""
name: m
model_id: org/m
model_type: text_embedding
params_billions: 1.0
embedding_dim: 768
supported_precisions:
  - fp16
""")
        experiment_yaml = tmp_path / "exp.yaml"
        experiment_yaml.write_text("""
name: exp
description: test
models:
  - m
precisions:
  - fp999
""")
        with pytest.raises(ValueError, match="precision"):
            load_experiment_config(experiment_yaml, models_dir)

    def test_experiment_config_pipeline_settings(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "m.yaml").write_text("""
name: m
model_id: org/m
model_type: text_embedding
params_billions: 1.0
embedding_dim: 768
supported_precisions:
  - fp16
""")
        experiment_yaml = tmp_path / "exp.yaml"
        experiment_yaml.write_text("""
name: exp
description: test
models:
  - m
precisions:
  - fp16
pipeline:
  chunk_size: 256
  chunk_overlap: 25
  retrieval_top_k: 5
  reranker_top_k: 3
""")
        config = load_experiment_config(experiment_yaml, models_dir)
        assert config.pipeline.chunk_size == 256
        assert config.pipeline.chunk_overlap == 25
        assert config.pipeline.retrieval_top_k == 5
        assert config.pipeline.reranker_top_k == 3

    def test_experiment_config_missing_model_raises(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        experiment_yaml = tmp_path / "exp.yaml"
        experiment_yaml.write_text("""
name: exp
description: test
models:
  - nonexistent-model
precisions:
  - fp16
""")
        with pytest.raises(ValueError, match="nonexistent-model"):
            load_experiment_config(experiment_yaml, models_dir)

    def test_experiment_config_missing_name_raises(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        experiment_yaml = tmp_path / "exp.yaml"
        experiment_yaml.write_text("description: test\nmodels: []\nprecisions:\n  - fp16\n")
        with pytest.raises(ValueError, match="name"):
            load_experiment_config(experiment_yaml, models_dir)

    def test_experiment_config_unknown_pipeline_field_raises(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        experiment_yaml = tmp_path / "exp.yaml"
        experiment_yaml.write_text("name: exp\ndescription: test\nmodels: []\nprecisions: []\npipeline:\n  invalid_field: 123\n")
        with pytest.raises(ValueError, match="Unknown pipeline"):
            load_experiment_config(experiment_yaml, models_dir)
