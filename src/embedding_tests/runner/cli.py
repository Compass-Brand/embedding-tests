"""CLI entry point for experiment execution."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from embedding_tests.config.datasets import list_all_datasets, load_dataset
from embedding_tests.config.experiment import load_experiment_config
from embedding_tests.config.models import load_all_model_configs
from embedding_tests.runner.experiment import ExperimentRunner

app = typer.Typer(name="emb-test", help="Embedding model testing framework")
console = Console()
logger = logging.getLogger(__name__)

# Navigate from src/embedding_tests/runner/cli.py to project root
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = Path(os.environ.get("EMB_TEST_CONFIGS_DIR", str(_PACKAGE_ROOT / "configs")))
MODELS_DIR = Path(os.environ.get("EMB_TEST_MODELS_DIR", str(CONFIGS_DIR / "models")))
RESULTS_DIR = Path(os.environ.get("EMB_TEST_RESULTS_DIR", str(_PACKAGE_ROOT / "results")))
DATA_DIR = Path(os.environ.get("EMB_TEST_DATA_DIR", str(_PACKAGE_ROOT / "data")))


@app.command()
def run(
    config: str = typer.Argument(..., help="Path to experiment config YAML"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Checkpoint directory"),
    clear_checkpoints: bool = typer.Option(
        False, "--clear-checkpoints",
        help="Clear checkpoints after successful completion",
    ),
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
    corpus, queries = load_dataset(dataset_name, data_dir=DATA_DIR)
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
        clear_on_success=clear_checkpoints,
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
def datasets(
    category: Optional[str] = typer.Option(
        None,
        "--category", "-c",
        help="Filter by category: nano, beir, code, technical, scientific",
    ),
) -> None:
    """List all available datasets."""
    try:
        all_datasets = list_all_datasets(category=category)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    if not all_datasets:
        console.print("[yellow]No datasets found[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Available Datasets{f' ({category})' if category else ''}")
    table.add_column("Name", style="cyan")
    table.add_column("Category")
    table.add_column("Description")

    for ds in all_datasets:
        table.add_row(
            ds["name"],
            ds.get("category", ""),
            ds.get("description", ""),
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(all_datasets)} datasets[/dim]")


@app.command()
def download(
    dataset: str = typer.Argument(..., help="Dataset name or 'all' to download all"),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory (default: data/)",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category", "-c",
        help="Category filter when using 'all': nano, beir, code, technical",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download even if already cached",
    ),
) -> None:
    """Pre-download datasets for offline use.

    Use 'all' as the dataset name to download all datasets, optionally
    filtered by category.

    Examples:
        emb-test download nfcorpus              # Single dataset
        emb-test download all                   # ALL datasets
        emb-test download all --category nano   # All NanoBEIR datasets
        emb-test download all --category beir   # All BEIR datasets
    """
    from embedding_tests.config.cache import ensure_cache_dir, get_cache_dir

    # Determine cache directory
    cache_dir = get_cache_dir()
    if output_dir:
        cache_dir = Path(output_dir) / "hf_cache"
    ensure_cache_dir(cache_dir)

    if dataset == "all":
        _download_all_datasets(cache_dir, category)
    else:
        _download_single_dataset(dataset, cache_dir)


def _download_single_dataset(dataset: str, cache_dir: Path) -> None:
    """Download a single dataset."""
    console.print(f"[cyan]Downloading dataset: {dataset}[/cyan]")

    try:
        corpus, queries = load_dataset(dataset, cache_dir=cache_dir)
        console.print(f"[green]Downloaded {dataset}[/green]")
        console.print(f"  Corpus: {len(corpus)} documents")
        console.print(f"  Queries: {len(queries)} queries")

        queries_with_qrels = sum(1 for q in queries if q.get("relevant_doc_ids"))
        console.print(f"  Queries with qrels: {queries_with_qrels}")

    except FileNotFoundError as e:
        console.print(f"[red]Dataset not found: {e}[/red]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        raise typer.Exit(1) from None


def _download_all_datasets(cache_dir: Path, category: Optional[str]) -> None:
    """Download all datasets, optionally filtered by category."""
    try:
        all_datasets = list_all_datasets(category=category)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    # Filter out 'sample' as it's local fixtures
    hf_datasets = [d for d in all_datasets if d["name"] != "sample"]

    if not hf_datasets:
        console.print("[yellow]No datasets to download[/yellow]")
        raise typer.Exit(0)

    console.print(
        f"[cyan]Downloading {len(hf_datasets)} datasets"
        f"{f' ({category})' if category else ''}[/cyan]"
    )

    success_count = 0
    fail_count = 0

    for ds in hf_datasets:
        ds_name = ds["name"]
        try:
            corpus, queries = load_dataset(ds_name, cache_dir=cache_dir)
            console.print(f"  [green]{ds_name}[/green]: {len(corpus)} docs, {len(queries)} queries")
            success_count += 1
        except Exception as e:
            console.print(f"  [red]{ds_name}[/red]: {e}")
            fail_count += 1

    console.print(f"\n[green]Downloaded {success_count} datasets[/green]")
    if fail_count > 0:
        console.print(f"[red]Failed: {fail_count} datasets[/red]")


@app.command()
def mteb(
    config: str = typer.Argument(..., help="Experiment config YAML"),
    tasks: Optional[str] = typer.Option(
        None,
        "--tasks", "-t",
        help="Comma-separated MTEB task names (overrides config)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for results",
    ),
) -> None:
    """Run MTEB benchmark evaluation.

    Uses MTEB's native evaluation framework for standard benchmarks.
    """
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        raise typer.Exit(1)

    experiment = load_experiment_config(config_path, MODELS_DIR)
    console.print(f"[green]Running MTEB benchmark: {experiment.name}[/green]")

    # Parse task names
    task_names: list[str] | None = None
    if tasks:
        task_names = [t.strip() for t in tasks.split(",")]
    elif hasattr(experiment, "mteb_tasks") and experiment.mteb_tasks:
        task_names = list(experiment.mteb_tasks)

    if not task_names:
        console.print("[red]No MTEB tasks specified. Use --tasks or add mteb_tasks to config.[/red]")
        raise typer.Exit(1)

    console.print(f"Tasks: {', '.join(task_names)}")
    console.print(f"Models: {len(experiment.models)}")

    from embedding_tests.config.hardware import detect_gpu
    from embedding_tests.evaluation.mteb_runner import MTEBModelAdapter, run_mteb_tasks
    from embedding_tests.hardware.precision import get_precision_config
    from embedding_tests.models.loader import load_model

    gpu = detect_gpu()
    if gpu is None:
        console.print("[red]No GPU detected. MTEB requires GPU.[/red]")
        raise typer.Exit(1)

    out_path = Path(output_dir) if output_dir else RESULTS_DIR / "mteb"
    out_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_config in experiment.models:
        for precision in experiment.precisions:
            if precision not in model_config.supported_precisions:
                continue

            console.print(f"\n[cyan]Evaluating {model_config.name} at {precision.value}[/cyan]")

            try:
                precision_config = get_precision_config(gpu, precision)
                model = load_model(model_config, precision_config)

                mteb_result = run_mteb_tasks(
                    model,
                    task_names=task_names,
                )

                all_results.append({
                    "model": model_config.name,
                    "precision": precision.value,
                    "tasks": mteb_result.get("tasks", []),
                    "results": mteb_result.get("results"),
                })

                model.unload()

            except Exception as e:
                logger.error("Failed %s/%s: %s", model_config.name, precision.value, e)
                all_results.append({
                    "model": model_config.name,
                    "precision": precision.value,
                    "error": str(e),
                })

    # Save results
    output_file = out_path / f"{experiment.name}_mteb.json"
    output_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    console.print(f"\n[green]Results saved to {output_file}[/green]")


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
