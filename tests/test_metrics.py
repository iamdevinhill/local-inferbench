"""Tests for the metrics module."""

from local_inferbench.adapters.base import GenerationResult
from local_inferbench.metrics import MetricsSummary, compute_metrics, _percentile


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 95) == 0.0

    def test_single_value(self):
        assert _percentile([5.0], 95) == 5.0

    def test_two_values(self):
        result = _percentile([1.0, 2.0], 50)
        assert abs(result - 1.5) < 0.01

    def test_p95(self):
        data = list(range(1, 101))
        result = _percentile([float(x) for x in data], 95)
        assert 95.0 <= result <= 96.0

    def test_p0_returns_min(self):
        assert _percentile([1.0, 2.0, 3.0], 0) == 1.0

    def test_p100_returns_max(self):
        assert _percentile([1.0, 2.0, 3.0], 100) == 3.0


class TestComputeMetrics:
    def test_empty_results(self):
        m = compute_metrics([])
        assert m.ttft_mean == 0.0
        assert m.total_tokens_generated == 0
        assert m.total_time == 0.0

    def test_single_result(self):
        results = [
            GenerationResult(
                text="hello",
                prompt_tokens=10,
                completion_tokens=20,
                time_to_first_token=0.1,
                total_time=1.0,
                finish_reason="stop",
            )
        ]
        m = compute_metrics(results)
        assert m.ttft_mean == 0.1
        assert m.ttft_median == 0.1
        assert m.tokens_per_second_mean == 20.0
        assert m.total_tokens_generated == 20
        assert m.total_time == 1.0

    def test_multiple_results(self, sample_generation_results):
        m = compute_metrics(sample_generation_results)

        assert m.ttft_min == 0.10
        assert m.ttft_max == 0.20
        assert 0.10 <= m.ttft_mean <= 0.20
        assert 0.10 <= m.ttft_median <= 0.20
        assert m.ttft_p95 >= m.ttft_median

        assert m.tokens_per_second_mean > 0
        assert m.tokens_per_second_median > 0
        assert m.total_tokens_generated == 50 + 80 + 100 + 30 + 60
        assert abs(m.total_time - (2.5 + 3.2 + 4.0 + 1.5 + 2.8)) < 0.001

    def test_prompt_eval_speed(self, sample_generation_results):
        m = compute_metrics(sample_generation_results)
        assert m.prompt_eval_speed > 0

    def test_zero_time_result_excluded_from_tps(self):
        results = [
            GenerationResult(
                text="",
                prompt_tokens=0,
                completion_tokens=0,
                time_to_first_token=0.1,
                total_time=0.0,
                finish_reason="stop",
            ),
            GenerationResult(
                text="hello",
                prompt_tokens=5,
                completion_tokens=10,
                time_to_first_token=0.2,
                total_time=1.0,
                finish_reason="stop",
            ),
        ]
        m = compute_metrics(results)
        assert m.tokens_per_second_mean == 10.0

    def test_to_dict(self, sample_generation_results):
        m = compute_metrics(sample_generation_results)
        d = m.to_dict()
        assert "ttft_mean" in d
        assert "tokens_per_second_mean" in d
        assert "total_tokens_generated" in d
        assert isinstance(d["total_tokens_generated"], int)
