"""Pydantic models for benchmark configuration."""

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    model_config = ConfigDict(strict=True)

    warmup_runs: int = Field(default=2, ge=0, description="Number of warmup generations to discard")
    profile: str = Field(default="quick", description="Prompt profile name or path to YAML file")
    max_tokens: int = Field(default=512, ge=1, description="Maximum tokens to generate per prompt")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    hardware_monitor: bool = Field(default=True, description="Enable hardware monitoring")
    hardware_poll_interval: float = Field(default=0.5, gt=0.0, description="Hardware poll interval in seconds")
    gpu_device: int | None = Field(default=None, description="GPU device index to monitor (None = default)")
    quality_scoring: bool = Field(default=True, description="Enable quality/accuracy scoring of responses")
