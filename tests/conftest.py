"""Shared fixtures for localbench tests."""

import pytest

from local_inferbench.adapters.base import GenerationResult


@pytest.fixture
def sample_generation_results() -> list[GenerationResult]:
    """A set of realistic generation results for testing metrics."""
    return [
        GenerationResult(
            text="Hello world",
            prompt_tokens=10,
            completion_tokens=50,
            time_to_first_token=0.15,
            total_time=2.5,
            finish_reason="stop",
        ),
        GenerationResult(
            text="Test output",
            prompt_tokens=15,
            completion_tokens=80,
            time_to_first_token=0.12,
            total_time=3.2,
            finish_reason="stop",
        ),
        GenerationResult(
            text="Another response",
            prompt_tokens=20,
            completion_tokens=100,
            time_to_first_token=0.18,
            total_time=4.0,
            finish_reason="stop",
        ),
        GenerationResult(
            text="Short",
            prompt_tokens=5,
            completion_tokens=30,
            time_to_first_token=0.10,
            total_time=1.5,
            finish_reason="stop",
        ),
        GenerationResult(
            text="Medium length response here",
            prompt_tokens=12,
            completion_tokens=60,
            time_to_first_token=0.20,
            total_time=2.8,
            finish_reason="length",
        ),
    ]


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a temporary database."""
    return str(tmp_path / "test_results.db")
