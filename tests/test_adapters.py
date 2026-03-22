"""Tests for adapter interface compliance."""

import pytest

from local_inferbench.adapters.base import BaseAdapter, GenerationResult
from local_inferbench.adapters.ollama import OllamaAdapter


class TestGenerationResult:
    def test_creation(self):
        r = GenerationResult(
            text="hello",
            prompt_tokens=5,
            completion_tokens=10,
            time_to_first_token=0.1,
            total_time=1.0,
            finish_reason="stop",
        )
        assert r.text == "hello"
        assert r.prompt_tokens == 5
        assert r.completion_tokens == 10
        assert r.time_to_first_token == 0.1
        assert r.total_time == 1.0
        assert r.finish_reason == "stop"


class TestOllamaAdapter:
    def test_name(self):
        adapter = OllamaAdapter(model="llama3:8b")
        assert adapter.name() == "ollama"

    def test_model_id(self):
        adapter = OllamaAdapter(model="llama3:8b")
        assert adapter.model_id() == "llama3:8b"

    def test_custom_base_url(self):
        adapter = OllamaAdapter(model="test", base_url="http://custom:1234/")
        assert adapter._base_url == "http://custom:1234"

    def test_metadata_returns_dict_on_error(self):
        adapter = OllamaAdapter(model="nonexistent", base_url="http://localhost:1")
        result = adapter.metadata()
        assert isinstance(result, dict)

    def test_is_base_adapter_subclass(self):
        assert issubclass(OllamaAdapter, BaseAdapter)
