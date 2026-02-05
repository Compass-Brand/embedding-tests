"""Tests for CLI entry point."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from embedding_tests.runner.cli import app


runner = CliRunner()


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_list_command_shows_models(self) -> None:
        from rich.console import Console

        wide_console = Console(width=200)
        with patch("embedding_tests.runner.cli.console", wide_console):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # Should list available models - check for full model name
        assert "qwen3-embedding-8b" in result.stdout.lower()

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
        mock_load.assert_called_once()
        call_args = mock_load.call_args[0][0]
        assert str(call_args).endswith("quick_sanity.yaml")

    def test_cli_report_command(self) -> None:
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0

    def test_cli_report_command_validates_directory(self) -> None:
        result = runner.invoke(app, ["report", "/nonexistent/path"])
        assert "no results found" in result.stdout.lower()
