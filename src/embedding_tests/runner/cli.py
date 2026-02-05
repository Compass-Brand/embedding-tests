"""CLI entry point for experiment execution."""

from __future__ import annotations

from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from embedding_tests.config.experiment import load_experiment_config
from embedding_tests.config.models import load_all_model_configs
from embedding_tests.runner.experiment import ExperimentRunner

app = typer.Typer(name="emb-test", help="Embedding model testing framework")
console = Console()

CONFIGS_DIR = Path("configs")
MODELS_DIR = CONFIGS_DIR / "models"
RESULTS_DIR = Path("results")


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

    runner = ExperimentRunner(
        model_configs=experiment.models,
        precisions=experiment.precisions,
        corpus=[],  # Will be loaded from datasets
        queries=[],
        checkpoint_dir=Path(checkpoint_dir),
        top_k=experiment.pipeline.retrieval_top_k,
        chunk_size=experiment.pipeline.chunk_size,
        chunk_overlap=experiment.pipeline.chunk_overlap,
    )
    results = runner.run()
    console.print(f"[green]Completed {len(results)} combinations[/green]")


@app.command(name="list")
def list_models() -> None:
    """List available models."""
    if not MODELS_DIR.exists():
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

    console.print(f"[green]Generating {output_format} report from {results_path}[/green]")


if __name__ == "__main__":
    app()
