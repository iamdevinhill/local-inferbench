"""Result classes for benchmark output, comparison, and export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from local_inferbench.adapters.base import GenerationResult
from local_inferbench.hardware import HardwareSummary
from local_inferbench.metrics import MetricsSummary
from local_inferbench.scoring import ScoringSummary


@dataclass
class BenchmarkResult:
    """Holds all data from a single benchmark run."""

    run_id: int | None
    adapter_name: str
    model_id: str
    profile: str
    config: dict[str, Any]
    metrics: MetricsSummary
    generation_results: list[GenerationResult]
    hardware_summary: HardwareSummary | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)
    scoring_summary: ScoringSummary | None = None

    def summary(self) -> None:
        """Print a rich-formatted summary to the terminal."""
        console = Console()
        m = self.metrics

        console.print()
        console.print(f"[bold cyan]Benchmark Results[/bold cyan]  —  {self.adapter_name}/{self.model_id}")
        if self.run_id is not None:
            console.print(f"  Run ID: {self.run_id}  |  Profile: {self.profile}")
        console.print()

        table = Table(title="Performance Metrics", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("TTFT mean", f"{m.ttft_mean:.3f}s")
        table.add_row("TTFT median", f"{m.ttft_median:.3f}s")
        table.add_row("TTFT p95", f"{m.ttft_p95:.3f}s")
        table.add_row("TTFT min", f"{m.ttft_min:.3f}s")
        table.add_row("TTFT max", f"{m.ttft_max:.3f}s")
        table.add_row("", "")
        table.add_row("Tokens/sec mean", f"{m.tokens_per_second_mean:.1f}")
        table.add_row("Tokens/sec median", f"{m.tokens_per_second_median:.1f}")
        table.add_row("Tokens/sec p95", f"{m.tokens_per_second_p95:.1f}")
        table.add_row("", "")
        table.add_row("Prompt eval speed", f"{m.prompt_eval_speed:.1f} tok/s")
        table.add_row("Total tokens", str(m.total_tokens_generated))
        table.add_row("Total time", f"{m.total_time:.2f}s")

        console.print(table)

        if self.hardware_summary:
            hw = self.hardware_summary
            hw_table = Table(title="Hardware", show_header=True)
            hw_table.add_column("Metric", style="bold")
            hw_table.add_column("Value", justify="right")

            if hw.gpu_name:
                hw_table.add_row("GPU", hw.gpu_name)
            if hw.peak_vram_gb is not None:
                hw_table.add_row("Peak VRAM", f"{hw.peak_vram_gb:.2f} GB")
            if hw.avg_gpu_utilization is not None:
                hw_table.add_row("Avg GPU util", f"{hw.avg_gpu_utilization:.1f}%")
            if hw.peak_gpu_temperature is not None:
                hw_table.add_row("Peak GPU temp", f"{hw.peak_gpu_temperature:.0f}°C")
            hw_table.add_row("Avg CPU", f"{hw.avg_cpu_percent:.1f}%")
            hw_table.add_row("Peak RAM", f"{hw.peak_ram_used_gb:.2f} GB")

            console.print(hw_table)

        if self.scoring_summary:
            ss = self.scoring_summary
            score_table = Table(title="Quality Scores", show_header=True)
            score_table.add_column("Metric", style="bold")
            score_table.add_column("Value", justify="right")

            score_table.add_row("Overall mean", f"{ss.mean_overall:.3f}")
            score_table.add_row("Overall median", f"{ss.median_overall:.3f}")
            score_table.add_row("Overall min", f"{ss.min_overall:.3f}")
            score_table.add_row("Overall max", f"{ss.max_overall:.3f}")
            score_table.add_row("", "")
            for cat, score in ss.mean_by_category.items():
                score_table.add_row(f"  {cat}", f"{score:.3f}")

            console.print(score_table)
        console.print()

    def export(self, path: str) -> None:
        """Export results to JSON or CSV based on file extension."""
        from local_inferbench.export import export_result

        export_result(self, path)

    def to_dataframe(self) -> Any:
        """Return a pandas DataFrame of generation results."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). Install with: pip install local-inferbench[pandas]"
            )

        rows = []
        for gr in self.generation_results:
            rows.append({
                "adapter": self.adapter_name,
                "model": self.model_id,
                "text_length": len(gr.text),
                "prompt_tokens": gr.prompt_tokens,
                "completion_tokens": gr.completion_tokens,
                "time_to_first_token": gr.time_to_first_token,
                "total_time": gr.total_time,
                "tokens_per_second": (
                    gr.completion_tokens / gr.total_time if gr.total_time > 0 else 0
                ),
                "finish_reason": gr.finish_reason,
            })
        return pd.DataFrame(rows)


@dataclass
class ComparisonResult:
    """Compares multiple BenchmarkResult objects side by side."""

    results: list[BenchmarkResult]

    def table(self) -> None:
        """Print a rich comparison table."""
        console = Console()
        t = Table(title="Benchmark Comparison", show_header=True)
        t.add_column("Metric", style="bold")
        for r in self.results:
            t.add_column(f"{r.adapter_name}/{r.model_id}", justify="right")

        metrics_rows = [
            ("TTFT mean", lambda m: f"{m.ttft_mean:.3f}s"),
            ("TTFT median", lambda m: f"{m.ttft_median:.3f}s"),
            ("TTFT p95", lambda m: f"{m.ttft_p95:.3f}s"),
            ("Tokens/sec mean", lambda m: f"{m.tokens_per_second_mean:.1f}"),
            ("Tokens/sec median", lambda m: f"{m.tokens_per_second_median:.1f}"),
            ("Total tokens", lambda m: str(m.total_tokens_generated)),
            ("Total time", lambda m: f"{m.total_time:.2f}s"),
        ]

        for label, fmt in metrics_rows:
            t.add_row(label, *[fmt(r.metrics) for r in self.results])

        # Add quality score if available
        if any(r.scoring_summary for r in self.results):
            t.add_row("", *["" for _ in self.results])
            t.add_row(
                "Quality score",
                *[
                    f"{r.scoring_summary.mean_overall:.3f}" if r.scoring_summary else "n/a"
                    for r in self.results
                ],
            )

        console.print(t)

    def fastest(self) -> BenchmarkResult:
        """Return the result with the highest mean tokens/sec."""
        return max(self.results, key=lambda r: r.metrics.tokens_per_second_mean)

    def most_efficient(self) -> BenchmarkResult:
        """Return the result with the lowest mean TTFT."""
        return min(self.results, key=lambda r: r.metrics.ttft_mean)
