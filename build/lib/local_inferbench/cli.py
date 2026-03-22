"""Click CLI entrypoint for local_inferbench."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.2.0", prog_name="inferbench")
def cli() -> None:
    """inferbench — benchmark Ollama models."""


@cli.command()
@click.option("--model", required=True, help="Ollama model name")
@click.option("--profile", default="quick", help="Benchmark profile name or YAML path")
@click.option("--warmup", default=2, type=int, help="Number of warmup runs")
@click.option("--max-tokens", default=512, type=int, help="Max tokens per generation")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature")
@click.option("--gpu-device", default=None, type=int, help="GPU device index to monitor")
@click.option("--base-url", default="http://localhost:11434", help="Ollama server URL")
@click.option("--db-path", default=None, help="Custom path for results database")
@click.option("--no-scoring", is_flag=True, help="Disable quality scoring")
def run(
    model: str,
    profile: str,
    warmup: int,
    max_tokens: int,
    temperature: float,
    gpu_device: int | None,
    base_url: str,
    db_path: str | None,
    no_scoring: bool,
) -> None:
    """Run a benchmark against an Ollama model."""
    from local_inferbench.benchmark import Benchmark
    from local_inferbench.config import BenchmarkConfig

    config = BenchmarkConfig(
        warmup_runs=warmup,
        profile=profile,
        max_tokens=max_tokens,
        temperature=temperature,
        gpu_device=gpu_device,
        quality_scoring=not no_scoring,
    )

    bench = Benchmark(models=[model], config=config, base_url=base_url, db_path=db_path)
    try:
        bench.run()
    finally:
        bench.close()


@cli.command()
@click.option("--last", default=None, type=int, help="Compare the last N runs")
@click.option("--runs", multiple=True, type=int, help="Specific run IDs to compare")
@click.option("--db-path", default=None, help="Custom path for results database")
def compare(last: int | None, runs: tuple[int, ...], db_path: str | None) -> None:
    """Compare benchmark runs side by side."""
    from local_inferbench.results import BenchmarkResult, ComparisonResult
    from local_inferbench.metrics import MetricsSummary
    from local_inferbench.storage import Storage

    storage = Storage(db_path)

    if last:
        run_list = storage.list_runs(limit=last)
        run_ids = [r["id"] for r in run_list]
    elif runs:
        run_ids = list(runs)
    else:
        console.print("[red]Specify --last N or --runs ID1 ID2 ...[/red]")
        return

    if len(run_ids) < 2:
        console.print("[red]Need at least 2 runs to compare.[/red]")
        return

    run_data = storage.get_comparison(run_ids)
    storage.close()

    results = []
    for rd in run_data:
        metrics = MetricsSummary(**rd["metrics"])
        results.append(BenchmarkResult(
            run_id=rd["id"],
            adapter_name=rd["adapter_name"],
            model_id=rd["model_id"],
            profile=rd["profile"],
            config=rd["config"] or {},
            metrics=metrics,
            generation_results=[],
            model_metadata=rd.get("model_metadata") or {},
        ))

    comparison = ComparisonResult(results=results)
    comparison.table()

    fastest = comparison.fastest()
    console.print(f"\n[bold green]Fastest:[/bold green] {fastest.adapter_name}/{fastest.model_id}")


@cli.command()
@click.option("--model", default=None, help="Filter by model name")
@click.option("--limit", default=20, type=int, help="Max results to show")
@click.option("--db-path", default=None, help="Custom path for results database")
def history(model: str | None, limit: int, db_path: str | None) -> None:
    """Show benchmark run history."""
    from local_inferbench.storage import Storage

    storage = Storage(db_path)
    runs = storage.list_runs(model=model, limit=limit)
    storage.close()

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title="Benchmark History", show_header=True)
    table.add_column("ID", justify="right")
    table.add_column("Timestamp")
    table.add_column("Model")
    table.add_column("Profile")

    for r in runs:
        table.add_row(
            str(r["id"]),
            r["timestamp"][:19],
            r["model_id"],
            r["profile"],
        )

    console.print(table)


@cli.command()
def profiles() -> None:
    """List available benchmark profiles."""
    from local_inferbench.prompts.registry import list_profiles

    profiles_list = list_profiles()

    table = Table(title="Available Profiles", show_header=True)
    table.add_column("Name", style="bold")
    table.add_column("Prompts", justify="right")
    table.add_column("Description")

    for p in profiles_list:
        table.add_row(p["name"], p["prompt_count"], p["description"])

    console.print(table)


@cli.command()
def info() -> None:
    """Show detected hardware and backend info."""
    from local_inferbench.hardware import detect_hardware

    hw = detect_hardware()

    console.print("\n[bold cyan]Hardware Info[/bold cyan]")
    console.print(f"  CPU cores (logical):  {hw['cpu_count']}")
    console.print(f"  CPU cores (physical): {hw['cpu_count_physical']}")
    console.print(f"  RAM total:            {hw['ram_total_gb']} GB")

    gpus = hw.get("gpus", [])
    if gpus:
        console.print(f"\n  [bold]GPUs:[/bold]")
        for gpu in gpus:
            console.print(f"    [{gpu['index']}] {gpu['name']} — {gpu['vram_total_gb']} GB")
    elif "gpu_monitoring" in hw:
        console.print(f"\n  GPU: {hw['gpu_monitoring']}")
    else:
        console.print("\n  GPU: none detected")

    console.print("\n[bold cyan]Backend: Ollama[/bold cyan]")
    console.print()


@cli.command(name="export")
@click.argument("run_id", type=int)
@click.option("--format", "fmt", type=click.Choice(["json", "csv"]), required=True, help="Export format")
@click.option("--output", required=True, help="Output file path")
@click.option("--db-path", default=None, help="Custom path for results database")
def export_cmd(run_id: int, fmt: str, output: str, db_path: str | None) -> None:
    """Export a benchmark run to JSON or CSV."""
    from local_inferbench.storage import Storage
    from local_inferbench.metrics import MetricsSummary
    from local_inferbench.adapters.base import GenerationResult
    from local_inferbench.results import BenchmarkResult

    storage = Storage(db_path)
    rd = storage.get_run(run_id)
    storage.close()

    if not rd:
        console.print(f"[red]Run {run_id} not found.[/red]")
        return

    metrics = MetricsSummary(**rd["metrics"])
    gen_results = [
        GenerationResult(**gr["result"])
        for gr in rd.get("generation_results", [])
    ]

    result = BenchmarkResult(
        run_id=rd["id"],
        adapter_name=rd["adapter_name"],
        model_id=rd["model_id"],
        profile=rd["profile"],
        config=rd["config"] or {},
        metrics=metrics,
        generation_results=gen_results,
        model_metadata=rd.get("model_metadata") or {},
    )

    if not output.endswith(f".{fmt}"):
        output = f"{output}.{fmt}"

    result.export(output)
    console.print(f"[green]Exported run {run_id} to {output}[/green]")


@cli.command()
@click.argument("run_id", type=int)
@click.option("--db-path", default=None, help="Custom path for results database")
def delete(run_id: int, db_path: str | None) -> None:
    """Delete a benchmark run."""
    from local_inferbench.storage import Storage

    storage = Storage(db_path)
    deleted = storage.delete_run(run_id)
    storage.close()

    if deleted:
        console.print(f"[green]Deleted run {run_id}.[/green]")
    else:
        console.print(f"[red]Run {run_id} not found.[/red]")


@cli.command()
@click.option("--base-url", default="http://localhost:11434", help="Ollama server URL")
@click.option("--show-all", is_flag=True, help="Show all models including too-large ones")
@click.option("--suggest-pull", is_flag=True, help="Also suggest models to download")
def recommend(base_url: str, show_all: bool, suggest_pull: bool) -> None:
    """Recommend models based on your hardware."""
    from local_inferbench.hardware import detect_hardware
    from local_inferbench.recommend import recommend_models, suggest_models_to_pull

    hw = detect_hardware()

    # Hardware summary
    console.print("\n[bold cyan]Your Hardware[/bold cyan]")
    console.print(f"  RAM: {hw['ram_total_gb']} GB")
    gpus = hw.get("gpus", [])
    if gpus:
        for gpu in gpus:
            console.print(f"  GPU: {gpu['name']} — {gpu['vram_total_gb']} GB VRAM")
    else:
        console.print("  GPU: none detected (CPU-only inference)")

    # Recommendations for installed models
    console.print("\n[bold cyan]Installed Models[/bold cyan]")
    recs = recommend_models(base_url=base_url, hardware=hw)

    if not recs:
        console.print("  [dim]No models found. Is Ollama running?[/dim]")
    else:
        table = Table(show_header=True)
        table.add_column("Model", style="bold")
        table.add_column("Params")
        table.add_column("Quant")
        table.add_column("Est. VRAM", justify="right")
        table.add_column("Tier")
        table.add_column("Notes")

        tier_style = {"optimal": "green", "feasible": "yellow", "too_large": "red"}

        for r in recs:
            if not show_all and r.tier == "too_large":
                continue
            style = tier_style.get(r.tier, "white")
            table.add_row(
                r.model_name,
                r.parameter_size,
                r.quantization,
                f"{r.estimated_vram_gb:.1f} GB",
                f"[{style}]{r.tier}[/{style}]",
                r.notes,
            )

        console.print(table)

    # Suggestions for models to pull
    if suggest_pull:
        installed_names = {r.model_name for r in recs}
        suggestions = suggest_models_to_pull(hardware=hw, installed_names=installed_names)

        if suggestions:
            console.print("\n[bold cyan]Suggested Models to Download[/bold cyan]")
            s_table = Table(show_header=True)
            s_table.add_column("Model", style="bold")
            s_table.add_column("Est. VRAM", justify="right")
            s_table.add_column("Tier")
            s_table.add_column("Notes")

            for s in suggestions:
                style = tier_style.get(s.tier, "white")
                s_table.add_row(
                    s.model_name,
                    f"{s.estimated_vram_gb:.1f} GB",
                    f"[{style}]{s.tier}[/{style}]",
                    s.notes,
                )

            console.print(s_table)
            console.print("\n  [dim]Install with: ollama pull <model_name>[/dim]")

    console.print()


@cli.command()
@click.argument("run_id", type=int)
@click.option("--db-path", default=None, help="Custom path for results database")
def score(run_id: int, db_path: str | None) -> None:
    """Score (or re-score) a past benchmark run for quality."""
    from local_inferbench.scoring import score_response, compute_scoring_summary, ScoringSummary
    from local_inferbench.storage import Storage

    storage = Storage(db_path)
    rd = storage.get_run(run_id)

    if not rd:
        console.print(f"[red]Run {run_id} not found.[/red]")
        storage.close()
        return

    gen_results = rd.get("generation_results", [])
    if not gen_results:
        console.print(f"[red]Run {run_id} has no generation results to score.[/red]")
        storage.close()
        return

    console.print(f"\n[bold cyan]Scoring run {run_id}[/bold cyan] — {rd['adapter_name']}/{rd['model_id']}")

    config = rd.get("config") or {}
    max_tokens = config.get("max_tokens", 512)

    response_scores = []
    for gr in gen_results:
        result = gr["result"]
        rs = score_response(
            prompt_text=gr["prompt_text"],
            prompt_category=gr["prompt_category"],
            response_text=result["text"],
            finish_reason=result["finish_reason"],
            max_tokens=max_tokens,
        )
        response_scores.append(rs)

    summary = compute_scoring_summary(response_scores)

    # Display results
    table = Table(title="Quality Scores", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Overall mean", f"{summary.mean_overall:.3f}")
    table.add_row("Overall median", f"{summary.median_overall:.3f}")
    table.add_row("Overall min", f"{summary.min_overall:.3f}")
    table.add_row("Overall max", f"{summary.max_overall:.3f}")
    table.add_row("", "")
    for cat, val in summary.mean_by_category.items():
        table.add_row(f"  {cat}", f"{val:.3f}")

    console.print(table)

    # Per-prompt breakdown
    detail_table = Table(title="Per-Prompt Scores", show_header=True)
    detail_table.add_column("#", justify="right")
    detail_table.add_column("Category")
    detail_table.add_column("Length", justify="right")
    detail_table.add_column("Coherence", justify="right")
    detail_table.add_column("Relevance", justify="right")
    detail_table.add_column("Complete", justify="right")
    detail_table.add_column("Category", justify="right")
    detail_table.add_column("Overall", justify="right")

    for i, rs in enumerate(response_scores, 1):
        detail_table.add_row(
            str(i),
            rs.prompt_category,
            f"{rs.length_score:.2f}",
            f"{rs.coherence_score:.2f}",
            f"{rs.relevance_score:.2f}",
            f"{rs.completeness_score:.2f}",
            f"{rs.category_score:.2f}",
            f"{rs.overall_score:.3f}",
        )

    console.print(detail_table)

    # Save to database
    storage._conn.execute(
        "INSERT OR REPLACE INTO quality_scores (run_id, summary) VALUES (?, ?)",
        (run_id, __import__("json").dumps(summary.to_dict())),
    )
    storage._conn.commit()
    storage.close()

    console.print(f"\n[green]Scores saved for run {run_id}.[/green]\n")
