"""Core Benchmark orchestrator — ties adapters, prompts, metrics, hardware, and storage together."""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.table import Table

from local_inferbench.adapters.base import BaseAdapter, GenerationResult
from local_inferbench.adapters.ollama import OllamaAdapter
from local_inferbench.config import BenchmarkConfig
from local_inferbench.hardware import HardwareMonitor, HardwareSummary
from local_inferbench.metrics import MetricsSummary, compute_metrics
from local_inferbench.prompts.registry import load_profile
from local_inferbench.results import BenchmarkResult
from local_inferbench.storage import Storage

logger = logging.getLogger(__name__)


class Benchmark:
    """Main orchestrator for running benchmarks."""

    def __init__(
        self,
        models: list[str],
        config: BenchmarkConfig | None = None,
        base_url: str = "http://localhost:11434",
        db_path: str | None = None,
    ) -> None:
        self.config = config or BenchmarkConfig()
        self._base_url = base_url
        self._adapters = [OllamaAdapter(model=m, base_url=base_url) for m in models]
        self._storage = Storage(db_path)
        self._console = Console()

    def run(self) -> list[BenchmarkResult]:
        """Run benchmarks across all models and return results."""
        profile = load_profile(self.config.profile)
        prompts = profile.prompts
        results: list[BenchmarkResult] = []

        for adapter in self._adapters:
            self._console.print(
                f"\n[bold green]Running benchmark:[/bold green] {adapter.name()}/{adapter.model_id()}"
            )
            self._console.print(f"  Profile: {profile.name} ({len(prompts)} prompts)")

            result = self._run_adapter(adapter, prompts, profile.name)
            results.append(result)
            result.summary()

        return results

    def _run_adapter(
        self,
        adapter: BaseAdapter,
        prompts: list[Any],
        profile_name: str,
    ) -> BenchmarkResult:
        """Run benchmark for a single adapter."""
        # Load model
        self._console.print("  Loading model...", style="dim")
        adapter.load()

        # Warmup
        if self.config.warmup_runs > 0:
            self._console.print(
                f"  Warming up ({self.config.warmup_runs} runs)...", style="dim"
            )
            for _ in range(self.config.warmup_runs):
                try:
                    adapter.generate(
                        "Hello",
                        max_tokens=16,
                        temperature=self.config.temperature,
                    )
                except Exception as e:
                    logger.warning("Warmup generation failed: %s", e)

        # Start hardware monitoring
        hw_monitor: HardwareMonitor | None = None
        hw_summary: HardwareSummary | None = None
        if self.config.hardware_monitor:
            hw_monitor = HardwareMonitor(
                poll_interval=self.config.hardware_poll_interval,
                gpu_device=self.config.gpu_device,
            )
            hw_monitor.start()

        # Run prompts
        gen_results: list[GenerationResult] = []
        prompt_records: list[dict[str, Any]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task("Generating", total=len(prompts))

            for prompt in prompts:
                try:
                    result = adapter.generate(
                        prompt.text,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )
                    gen_results.append(result)
                    prompt_records.append({
                        "prompt_text": prompt.text,
                        "prompt_category": prompt.category,
                        "result": {
                            "text": result.text,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "time_to_first_token": result.time_to_first_token,
                            "total_time": result.total_time,
                            "finish_reason": result.finish_reason,
                        },
                    })

                    tps = (
                        result.completion_tokens / result.total_time
                        if result.total_time > 0
                        else 0
                    )
                    progress.update(
                        task,
                        advance=1,
                        description=f"Generating ({tps:.1f} tok/s)",
                    )
                except Exception as e:
                    logger.error("Generation failed for prompt: %s", e)
                    progress.update(task, advance=1)

        # Stop hardware monitoring
        if hw_monitor:
            hw_monitor.stop()
            hw_summary = hw_monitor.summarize()

        # Compute metrics
        metrics = compute_metrics(gen_results)

        # Quality scoring
        scoring_summary = None
        if self.config.quality_scoring and gen_results:
            from local_inferbench.scoring import score_response, compute_scoring_summary

            self._console.print("  Scoring response quality...", style="dim")
            response_scores = []
            for prompt, gen_result in zip(prompts, gen_results):
                rs = score_response(
                    prompt_text=prompt.text,
                    prompt_category=prompt.category,
                    response_text=gen_result.text,
                    finish_reason=gen_result.finish_reason,
                    max_tokens=self.config.max_tokens,
                )
                response_scores.append(rs)
            scoring_summary = compute_scoring_summary(response_scores)

        # Get model metadata
        try:
            model_metadata = adapter.metadata()
        except Exception:
            model_metadata = {}

        # Unload model
        try:
            adapter.unload()
        except Exception as e:
            logger.warning("Failed to unload model: %s", e)

        # Store results
        config_dict = self.config.model_dump()
        run_id = self._storage.save_run(
            adapter_name=adapter.name(),
            model_id=adapter.model_id(),
            profile=profile_name,
            config=config_dict,
            generation_results=prompt_records,
            metrics=metrics.to_dict(),
            model_metadata=model_metadata,
            hardware_summary=hw_summary.to_dict() if hw_summary else None,
            scoring_summary=scoring_summary.to_dict() if scoring_summary else None,
        )

        return BenchmarkResult(
            run_id=run_id,
            adapter_name=adapter.name(),
            model_id=adapter.model_id(),
            profile=profile_name,
            config=config_dict,
            metrics=metrics,
            generation_results=gen_results,
            hardware_summary=hw_summary,
            model_metadata=model_metadata,
            scoring_summary=scoring_summary,
        )

    def close(self) -> None:
        """Close the storage connection."""
        self._storage.close()
