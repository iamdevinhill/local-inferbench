"""Metric calculation from generation results.

Uses only the stdlib statistics module — no numpy dependency.
"""

import statistics
from dataclasses import dataclass

from local_inferbench.adapters.base import GenerationResult


@dataclass
class MetricsSummary:
    """Aggregated metrics computed from a list of GenerationResult objects."""

    ttft_mean: float
    ttft_median: float
    ttft_p95: float
    ttft_min: float
    ttft_max: float

    tokens_per_second_mean: float
    tokens_per_second_median: float
    tokens_per_second_p95: float

    prompt_eval_speed: float

    total_tokens_generated: int
    total_time: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ttft_mean": self.ttft_mean,
            "ttft_median": self.ttft_median,
            "ttft_p95": self.ttft_p95,
            "ttft_min": self.ttft_min,
            "ttft_max": self.ttft_max,
            "tokens_per_second_mean": self.tokens_per_second_mean,
            "tokens_per_second_median": self.tokens_per_second_median,
            "tokens_per_second_p95": self.tokens_per_second_p95,
            "prompt_eval_speed": self.prompt_eval_speed,
            "total_tokens_generated": self.total_tokens_generated,
            "total_time": self.total_time,
        }


def _percentile(data: list[float], p: float) -> float:
    """Calculate the p-th percentile of a sorted list (0-100 scale)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def compute_metrics(results: list[GenerationResult]) -> MetricsSummary:
    """Compute aggregate metrics from a list of generation results."""
    if not results:
        return MetricsSummary(
            ttft_mean=0.0,
            ttft_median=0.0,
            ttft_p95=0.0,
            ttft_min=0.0,
            ttft_max=0.0,
            tokens_per_second_mean=0.0,
            tokens_per_second_median=0.0,
            tokens_per_second_p95=0.0,
            prompt_eval_speed=0.0,
            total_tokens_generated=0,
            total_time=0.0,
        )

    ttft_values = [r.time_to_first_token for r in results]
    tokens_per_sec_values = [
        r.completion_tokens / r.total_time
        for r in results
        if r.total_time > 0 and r.completion_tokens > 0
    ]

    total_prompt_tokens = sum(r.prompt_tokens for r in results)
    total_ttft = sum(r.time_to_first_token for r in results)
    prompt_eval_speed = total_prompt_tokens / total_ttft if total_ttft > 0 else 0.0

    total_tokens = sum(r.completion_tokens for r in results)
    total_time = sum(r.total_time for r in results)

    return MetricsSummary(
        ttft_mean=statistics.mean(ttft_values),
        ttft_median=statistics.median(ttft_values),
        ttft_p95=_percentile(ttft_values, 95),
        ttft_min=min(ttft_values),
        ttft_max=max(ttft_values),
        tokens_per_second_mean=(
            statistics.mean(tokens_per_sec_values) if tokens_per_sec_values else 0.0
        ),
        tokens_per_second_median=(
            statistics.median(tokens_per_sec_values) if tokens_per_sec_values else 0.0
        ),
        tokens_per_second_p95=(
            _percentile(tokens_per_sec_values, 95) if tokens_per_sec_values else 0.0
        ),
        prompt_eval_speed=prompt_eval_speed,
        total_tokens_generated=total_tokens,
        total_time=total_time,
    )
