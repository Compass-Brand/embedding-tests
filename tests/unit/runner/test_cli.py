"""Tests for CLI entry point."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from embedding_tests.runner.cli import app


runner = CliRunner()


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_list_command_shows_models(self) -> None:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # Should list available models
        assert "model" in result.stdout.lower() or "Model" in result.stdout

    @patch("embedding_tests.runner.cli.ExperimentRunner")
    @patch("embedding_tests.runner.cli.load_experiment_config")
    def test_cli_run_command_accepts_config_path(
        self, mock_load: MagicMock, mock_runner: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.models = []
        mock_config.precisions = []
        mock_config.pipeline = MagicMock()
        mock_load.return_value = mock_config
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = []
        mock_runner.return_value = mock_runner_instance

        result = runner.invoke(app, ["run", "configs/experiments/quick_sanity.yaml"])
        assert result.exit_code == 0

    def test_cli_report_command(self) -> None:
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0
