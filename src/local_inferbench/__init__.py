"""local_inferbench — a benchmarking framework for Ollama models."""

from local_inferbench.benchmark import Benchmark
from local_inferbench.config import BenchmarkConfig
from local_inferbench.adapters import OllamaAdapter
from local_inferbench.results import BenchmarkResult, ComparisonResult
from local_inferbench.recommend import recommend_models, suggest_models_to_pull
from local_inferbench.scoring import score_response, compute_scoring_summary

__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "OllamaAdapter",
    "BenchmarkResult",
    "ComparisonResult",
    "recommend_models",
    "suggest_models_to_pull",
    "score_response",
    "compute_scoring_summary",
]

__version__ = "0.2.1"
