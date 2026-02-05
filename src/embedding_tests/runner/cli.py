"""CLI entry point for experiment execution."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from embedding_tests.config.datasets import load_dataset
from embedding_tests.config.experiment import load_experiment_config
from embedding_tests.config.models import load_all_model_configs
from embedding_tests.runner.experiment import ExperimentRunner

app = typer.Typer(name="emb-test", help="Embedding model testing framework")
console = Console()

# Navigate from src/embedding_tests/runner/cli.py to project root
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = Path(os.environ.get("EMB_TEST_CONFIGS_DIR", str(_PACKAGE_ROOT / "configs")))
MODELS_DIR = Path(os.environ.get("EMB_TEST_MODELS_DIR", str(CONFIGS_DIR / "models")))
RESULTS_DIR = Path(os.environ.get("EMB_TEST_RESULTS_DIR", str(_PACKAGE_ROOT / "results")))


@app.command()
def run(
    config: str = typer.Argument(..., help="Path to experiment config YAML"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Checkpoint directory"),
) -> None:
    """Run an experiment from a config file."""
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        raise typer.Exit(1)

    experiment = load_experiment_config(config_path, MODELS_DIR)
    console.print(f"[green]Running experiment: {experiment.name}[/green]")
    console.print(f"Models: {len(experiment.models)}, Precisions: {len(experiment.precisions)}")

    # Load dataset(s) - use first specified or default to sample
    dataset_name = experiment.datasets[0] if experiment.datasets else None
    data_dir = _PACKAGE_ROOT / "data"
    corpus, queries = load_dataset(dataset_name, data_dir=data_dir)
    console.print(f"Dataset: {dataset_name or 'sample'} ({len(corpus)} docs, {len(queries)} queries)")

    runner = ExperimentRunner(
        model_configs=experiment.models,
        precisions=experiment.precisions,
        corpus=corpus,
        queries=queries,
        checkpoint_dir=Path(checkpoint_dir),
        top_k=experiment.pipeline.retrieval_top_k,
        chunk_size=experiment.pipeline.chunk_size,
        chunk_overlap=experiment.pipeline.chunk_overlap,
    )
    results = runner.run()
    console.print(f"[green]Completed {len(results)} combinations[/green]")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{experiment.name}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"Results saved to {output_path}")


@app.command(name="list")
def list_models() -> None:
    """List available models."""
    if not MODELS_DIR.is_dir():
        console.print("[yellow]No model configs found[/yellow]")
        raise typer.Exit(0)

    configs = load_all_model_configs(MODELS_DIR)
    table = Table(title="Available Models")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Params (B)")
    table.add_column("Embed Dim")
    table.add_column("Precisions")

    for cfg in configs:
        precisions = ", ".join(p.value for p in cfg.supported_precisions)
        table.add_row(
            cfg.name,
            cfg.model_type.value,
            f"{cfg.params_billions:.1f}",
            str(cfg.embedding_dim),
            precisions,
        )

    console.print(table)


@app.command()
def report(
    results_dir: str = typer.Argument("results", help="Results directory"),
    output_format: str = typer.Option("markdown", help="Output format (json, csv, markdown)"),
) -> None:
    """Generate reports from experiment results."""
    results_path = Path(results_dir)
    if not results_path.exists():
        console.print(f"[yellow]No results found at {results_path}[/yellow]")
        raise typer.Exit(0)

    # Load all JSON result files
    json_files = sorted(results_path.glob("*.json"))
    if not json_files:
        console.print(f"[yellow]No result files in {results_path}[/yellow]")
        raise typer.Exit(0)

    from embedding_tests.reporting.collector import ModelResult, ResultsCollector
    from embedding_tests.reporting.export import export_csv, export_json, export_markdown

    collector = ResultsCollector()
    for jf in json_files:
        raw = json.loads(jf.read_text(encoding="utf-8"))
        for entry in raw:
            result = _entry_to_model_result(entry)
            if result is not None:
                collector.add(result)

    if not collector.results:
        console.print("[yellow]No valid results to report[/yellow]")
        raise typer.Exit(0)

    output_dir = results_path / "reports"
    exporters = {
        "json": (export_json, "report.json"),
        "csv": (export_csv, "report.csv"),
        "markdown": (export_markdown, "report.md"),
    }

    if output_format not in exporters:
        console.print(f"[red]Unknown format: {output_format}. Use json, csv, or markdown.[/red]")
        raise typer.Exit(1)

    export_fn, filename = exporters[output_format]
    output_path = output_dir / filename
    export_fn(collector.results, output_path)
    console.print(f"[green]Report saved to {output_path}[/green]")


def _entry_to_model_result(entry: dict) -> ModelResult | None:
    """Convert a raw experiment result dict to a ModelResult."""
    from embedding_tests.reporting.collector import ModelResult

    if "error" in entry and entry.get("status") != "completed":
        return ModelResult(
            model_name=entry.get("model", "unknown"),
            precision=entry.get("precision", "unknown"),
            recall_at_10=0.0,
            mrr=0.0,
            ndcg_at_10=0.0,
            precision_at_10=0.0,
            total_time_seconds=0.0,
            error=entry.get("error"),
        )

    results = entry.get("results")
    if not results:
        return None

    # Average per-query metrics
    recalls = []
    precisions = []
    ndcgs = []
    for qid, metrics in results.items():
        for key, val in metrics.items():
            if "recall" in key:
                recalls.append(val)
            elif "precision" in key:
                precisions.append(val)
            elif "ndcg" in key:
                ndcgs.append(val)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    # MRR is stored at the top level (computed across all queries)
    mrr_score = entry.get("mrr", 0.0)

    return ModelResult(
        model_name=entry.get("model", "unknown"),
        precision=entry.get("precision", "unknown"),
        recall_at_10=avg_recall,
        mrr=mrr_score,
        ndcg_at_10=avg_ndcg,
        precision_at_10=avg_precision,
        total_time_seconds=entry.get("total_time", 0.0),
    )


if __name__ == "__main__":
    app()
