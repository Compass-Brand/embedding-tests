"""Tests for CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path
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

    @patch("embedding_tests.runner.cli.load_dataset")
    @patch("embedding_tests.runner.cli.ExperimentRunner")
    @patch("embedding_tests.runner.cli.load_experiment_config")
    def test_cli_run_command_accepts_config_path(
        self,
        mock_load: MagicMock,
        mock_runner: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.models = []
        mock_config.precisions = []
        mock_config.datasets = ["sample"]
        mock_config.pipeline = MagicMock()
        mock_config.name = "test_experiment"
        mock_load.return_value = mock_config
        mock_dataset.return_value = (
            [{"doc_id": "d1", "text": "doc"}],
            [{"query_id": "q1", "text": "query"}],
        )
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = []
        mock_runner.return_value = mock_runner_instance

        result = runner.invoke(app, ["run", "configs/experiments/quick_sanity.yaml"])
        assert result.exit_code == 0
        mock_load.assert_called_once()
        call_args = mock_load.call_args[0][0]
        assert str(call_args).endswith("quick_sanity.yaml")
        mock_dataset.assert_called_once()

    @patch("embedding_tests.runner.cli.load_dataset")
    @patch("embedding_tests.runner.cli.ExperimentRunner")
    @patch("embedding_tests.runner.cli.load_experiment_config")
    def test_cli_run_command_loads_dataset_from_experiment(
        self,
        mock_load: MagicMock,
        mock_runner: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.models = []
        mock_config.precisions = []
        mock_config.datasets = ["custom_ds"]
        mock_config.pipeline = MagicMock()
        mock_config.name = "test_experiment"
        mock_load.return_value = mock_config
        mock_dataset.return_value = ([], [])
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = []
        mock_runner.return_value = mock_runner_instance

        result = runner.invoke(app, ["run", "configs/experiments/quick_sanity.yaml"])
        assert result.exit_code == 0
        mock_dataset.assert_called_once_with("custom_ds", data_dir=mock_dataset.call_args[1]["data_dir"])

    @patch("embedding_tests.runner.cli.load_dataset")
    @patch("embedding_tests.runner.cli.ExperimentRunner")
    @patch("embedding_tests.runner.cli.load_experiment_config")
    def test_cli_run_defaults_to_sample_dataset_when_empty(
        self,
        mock_load: MagicMock,
        mock_runner: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.models = []
        mock_config.precisions = []
        mock_config.datasets = []
        mock_config.pipeline = MagicMock()
        mock_config.name = "test_experiment"
        mock_load.return_value = mock_config
        mock_dataset.return_value = ([], [])
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = []
        mock_runner.return_value = mock_runner_instance

        result = runner.invoke(app, ["run", "configs/experiments/quick_sanity.yaml"])
        assert result.exit_code == 0
        # When datasets is empty, should pass None (defaults to sample)
        mock_dataset.assert_called_once()
        assert mock_dataset.call_args[0][0] is None

    def test_cli_run_nonexistent_config_exits_1(self) -> None:
        result = runner.invoke(app, ["run", "/nonexistent/config.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    @patch("embedding_tests.runner.cli.load_dataset")
    @patch("embedding_tests.runner.cli.ExperimentRunner")
    @patch("embedding_tests.runner.cli.load_experiment_config")
    def test_cli_run_clear_checkpoints_flag(
        self,
        mock_load: MagicMock,
        mock_runner: MagicMock,
        mock_dataset: MagicMock,
    ) -> None:
        """Test that --clear-checkpoints flag is passed to ExperimentRunner."""
        mock_config = MagicMock()
        mock_config.models = []
        mock_config.precisions = []
        mock_config.datasets = ["sample"]
        mock_config.pipeline = MagicMock()
        mock_config.name = "test_experiment"
        mock_load.return_value = mock_config
        mock_dataset.return_value = ([], [])
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = []
        mock_runner.return_value = mock_runner_instance

        result = runner.invoke(app, ["run", "configs/experiments/quick_sanity.yaml", "--clear-checkpoints"])
        assert result.exit_code == 0
        # Verify clear_on_success=True was passed
        assert mock_runner.call_args.kwargs.get("clear_on_success") is True

    def test_cli_report_command_help(self) -> None:
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0

    def test_cli_report_command_validates_directory(self) -> None:
        result = runner.invoke(app, ["report", "/nonexistent/path"])
        assert "no results found" in result.stdout.lower()

    def test_cli_report_with_results(self, tmp_path: Path) -> None:
        results_data = [
            {
                "model": "test-model",
                "precision": "fp16",
                "status": "completed",
                "total_time": 10.5,
                "results": {
                    "q1": {"recall@10": 0.8, "precision@10": 0.6},
                    "q2": {"recall@10": 0.9, "precision@10": 0.7},
                },
            }
        ]
        result_file = tmp_path / "test.json"
        result_file.write_text(json.dumps(results_data))

        result = runner.invoke(
            app, ["report", str(tmp_path), "--output-format", "json"]
        )
        assert result.exit_code == 0
        assert "report saved" in result.stdout.lower()

        report_path = tmp_path / "reports" / "report.json"
        assert report_path.exists()
        report_data = json.loads(report_path.read_text())
        assert len(report_data) == 1
        assert report_data[0]["model_name"] == "test-model"

    def test_cli_report_with_error_results(self, tmp_path: Path) -> None:
        results_data = [
            {
                "model": "broken-model",
                "precision": "fp16",
                "status": "failed",
                "error": "OOM error",
            }
        ]
        result_file = tmp_path / "errors.json"
        result_file.write_text(json.dumps(results_data))

        result = runner.invoke(
            app, ["report", str(tmp_path), "--output-format", "json"]
        )
        assert result.exit_code == 0
        report_path = tmp_path / "reports" / "report.json"
        assert report_path.exists()
        report_data = json.loads(report_path.read_text())
        assert len(report_data) == 1
        assert report_data[0]["error"] == "OOM error"
        assert report_data[0]["recall_at_10"] == 0.0

    def test_cli_report_empty_results_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["report", str(tmp_path)])
        assert "no result files" in result.stdout.lower()

    def test_cli_report_unknown_format(self, tmp_path: Path) -> None:
        result_file = tmp_path / "test.json"
        result_file.write_text(json.dumps([{"model": "x", "results": {"q1": {"recall@10": 0.5}}}]))
        result = runner.invoke(
            app, ["report", str(tmp_path), "--output-format", "xml"]
        )
        assert result.exit_code == 1
        assert "unknown format" in result.stdout.lower()

    def test_cli_report_markdown_format(self, tmp_path: Path) -> None:
        results_data = [
            {
                "model": "model-a",
                "precision": "fp16",
                "status": "completed",
                "total_time": 5.0,
                "results": {
                    "q1": {"recall@10": 0.7, "precision@10": 0.5},
                },
            }
        ]
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(results_data))

        result = runner.invoke(
            app, ["report", str(tmp_path), "--output-format", "markdown"]
        )
        assert result.exit_code == 0
        report_path = tmp_path / "reports" / "report.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "model-a" in content
        assert "recall@10" in content


class TestCLIDatasets:
    """Tests for datasets command."""

    def test_datasets_command_lists_all(self) -> None:
        from rich.console import Console

        wide_console = Console(width=200)
        with patch("embedding_tests.runner.cli.console", wide_console):
            result = runner.invoke(app, ["datasets"])
        assert result.exit_code == 0
        assert "sample" in result.stdout
        assert "nano-nfcorpus" in result.stdout
        assert "nfcorpus" in result.stdout
        assert "codesearchnet-python" in result.stdout

    def test_datasets_command_filter_by_category(self) -> None:
        from rich.console import Console

        wide_console = Console(width=200)
        with patch("embedding_tests.runner.cli.console", wide_console):
            result = runner.invoke(app, ["datasets", "--category", "nano"])
        assert result.exit_code == 0
        assert "nano-nfcorpus" in result.stdout
        # Should NOT have non-nano datasets
        assert "codesearchnet-python" not in result.stdout

    def test_datasets_command_invalid_category(self) -> None:
        result = runner.invoke(app, ["datasets", "--category", "invalid"])
        assert result.exit_code == 1
        assert "unknown category" in result.stdout.lower()


class TestCLIDownload:
    """Tests for download command."""

    @patch("embedding_tests.runner.cli.load_dataset")
    def test_download_command_success(self, mock_load: MagicMock) -> None:
        mock_load.return_value = (
            [{"doc_id": "d1", "text": "doc"}] * 100,
            [{"query_id": "q1", "text": "q", "relevant_doc_ids": ["d1"]}] * 10,
        )
        result = runner.invoke(app, ["download", "nano-nfcorpus"])
        assert result.exit_code == 0
        assert "downloaded" in result.stdout.lower()
        assert "100 documents" in result.stdout.lower()
        assert "10 queries" in result.stdout.lower()

    def test_download_command_not_found(self) -> None:
        # This will trigger the local dataset path which doesn't exist
        result = runner.invoke(app, ["download", "totally_fake_dataset"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    @patch("embedding_tests.runner.cli.load_dataset")
    def test_download_all_with_category(self, mock_load: MagicMock) -> None:
        """Test downloading all datasets in a category."""
        mock_load.return_value = (
            [{"doc_id": "d1", "text": "doc"}],
            [{"query_id": "q1", "text": "q", "relevant_doc_ids": ["d1"]}],
        )
        result = runner.invoke(app, ["download", "all", "--category", "nano"])
        assert result.exit_code == 0
        # Should download all 6 nano datasets
        assert mock_load.call_count == 6
        assert "nano-nfcorpus" in result.stdout

    @patch("embedding_tests.runner.cli.load_dataset")
    def test_download_all_downloads_all_datasets(self, mock_load: MagicMock) -> None:
        """Test downloading all datasets without category filter."""
        mock_load.return_value = (
            [{"doc_id": "d1", "text": "doc"}],
            [{"query_id": "q1", "text": "q", "relevant_doc_ids": ["d1"]}],
        )
        result = runner.invoke(app, ["download", "all"])
        assert result.exit_code == 0
        # Should download many datasets (34 total - 1 sample = 33 HF datasets)
        assert mock_load.call_count >= 30

    def test_download_invalid_category(self) -> None:
        """Test error on invalid category."""
        result = runner.invoke(app, ["download", "all", "--category", "invalid"])
        assert result.exit_code == 1
        assert "unknown category" in result.stdout.lower()


class TestCLIMTEB:
    """Tests for mteb command."""

    def test_mteb_command_missing_config(self) -> None:
        result = runner.invoke(app, ["mteb", "/nonexistent/config.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    @patch("embedding_tests.runner.cli.load_experiment_config")
    def test_mteb_command_no_tasks_specified(self, mock_load: MagicMock) -> None:
        mock_config = MagicMock()
        mock_config.name = "test"
        mock_config.models = []
        mock_config.precisions = []
        mock_config.mteb_tasks = []
        mock_load.return_value = mock_config

        result = runner.invoke(app, ["mteb", "configs/experiments/quick_sanity.yaml"])
        assert result.exit_code == 1
        assert "no mteb tasks" in result.stdout.lower()
