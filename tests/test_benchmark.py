"""Tests for the benchmark orchestrator."""

import pytest
from unittest.mock import MagicMock, patch

from local_inferbench.adapters.base import BaseAdapter, GenerationResult
from local_inferbench.benchmark import Benchmark
from local_inferbench.config import BenchmarkConfig


class MockAdapter(BaseAdapter):
    """A mock adapter for testing the benchmark orchestrator."""

    def __init__(self):
        self._loaded = False
        self._call_count = 0

    def name(self) -> str:
        return "mock"

    def model_id(self) -> str:
        return "mock-model"

    def load(self) -> None:
        self._loaded = True

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, **kwargs) -> GenerationResult:
        self._call_count += 1
        return GenerationResult(
            text=f"Response to: {prompt[:20]}",
            prompt_tokens=10,
            completion_tokens=20,
            time_to_first_token=0.05,
            total_time=0.5,
            finish_reason="stop",
        )

    def unload(self) -> None:
        self._loaded = False

    def metadata(self) -> dict:
        return {"type": "mock"}


class TestBenchmarkConfig:
    def test_defaults(self):
        config = BenchmarkConfig()
        assert config.warmup_runs == 2
        assert config.profile == "quick"
        assert config.max_tokens == 512
        assert config.temperature == 0.0
        assert config.hardware_monitor is True
        assert config.hardware_poll_interval == 0.5
        assert config.gpu_device is None

    def test_custom_values(self):
        config = BenchmarkConfig(
            warmup_runs=0,
            profile="standard",
            max_tokens=256,
            temperature=0.7,
            hardware_monitor=False,
        )
        assert config.warmup_runs == 0
        assert config.profile == "standard"
        assert config.max_tokens == 256


class TestBenchmark:
    def test_run_with_mock_adapter(self, tmp_db):
        mock_adapter = MockAdapter()
        config = BenchmarkConfig(
            warmup_runs=1,
            profile="quick",
            hardware_monitor=False,
        )
        with patch("local_inferbench.benchmark.OllamaAdapter", return_value=mock_adapter):
            bench = Benchmark(models=["mock-model"], config=config, db_path=tmp_db)
        results = bench.run()
        bench.close()

        assert len(results) == 1
        result = results[0]
        assert result.adapter_name == "mock"
        assert result.model_id == "mock-model"
        assert result.profile == "quick"
        assert result.run_id is not None
        assert len(result.generation_results) == 10  # quick profile has 10 prompts
        assert result.metrics.total_tokens_generated > 0

    def test_run_stores_in_db(self, tmp_db):
        from local_inferbench.storage import Storage

        mock_adapter = MockAdapter()
        config = BenchmarkConfig(warmup_runs=0, profile="quick", hardware_monitor=False)
        with patch("local_inferbench.benchmark.OllamaAdapter", return_value=mock_adapter):
            bench = Benchmark(models=["mock-model"], config=config, db_path=tmp_db)
        results = bench.run()
        bench.close()

        storage = Storage(tmp_db)
        runs = storage.list_runs()
        assert len(runs) == 1
        assert runs[0]["adapter_name"] == "mock"
        storage.close()

    def test_run_multiple_models(self, tmp_db):
        mock_adapter = MockAdapter()
        config = BenchmarkConfig(warmup_runs=0, profile="quick", hardware_monitor=False)
        with patch("local_inferbench.benchmark.OllamaAdapter", return_value=mock_adapter):
            bench = Benchmark(models=["model-a", "model-b"], config=config, db_path=tmp_db)
        results = bench.run()
        bench.close()

        assert len(results) == 2
