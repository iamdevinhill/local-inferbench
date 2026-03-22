"""JSON and CSV export utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_inferbench.results import BenchmarkResult


def export_result(result: BenchmarkResult, path: str) -> None:
    """Export a BenchmarkResult to JSON or CSV based on file extension."""
    if path.endswith(".json"):
        _export_json(result, path)
    elif path.endswith(".csv"):
        _export_csv(result, path)
    else:
        raise ValueError(f"Unsupported export format: {path}. Use .json or .csv")


def _export_json(result: BenchmarkResult, path: str) -> None:
    data = {
        "run_id": result.run_id,
        "adapter_name": result.adapter_name,
        "model_id": result.model_id,
        "profile": result.profile,
        "config": result.config,
        "model_metadata": result.model_metadata,
        "metrics": result.metrics.to_dict(),
        "hardware_summary": result.hardware_summary.to_dict() if result.hardware_summary else None,
        "generation_results": [asdict(gr) for gr in result.generation_results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _export_csv(result: BenchmarkResult, path: str) -> None:
    if not result.generation_results:
        return
    fieldnames = [
        "adapter",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "time_to_first_token",
        "total_time",
        "tokens_per_second",
        "finish_reason",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for gr in result.generation_results:
            writer.writerow({
                "adapter": result.adapter_name,
                "model": result.model_id,
                "prompt_tokens": gr.prompt_tokens,
                "completion_tokens": gr.completion_tokens,
                "time_to_first_token": gr.time_to_first_token,
                "total_time": gr.total_time,
                "tokens_per_second": (
                    gr.completion_tokens / gr.total_time if gr.total_time > 0 else 0
                ),
                "finish_reason": gr.finish_reason,
            })
