"""Tests for the SQLite storage layer."""

import pytest

from local_inferbench.storage import Storage


@pytest.fixture
def storage(tmp_db):
    s = Storage(tmp_db)
    yield s
    s.close()


@pytest.fixture
def sample_run_data():
    return {
        "adapter_name": "ollama",
        "model_id": "llama3:8b",
        "profile": "quick",
        "config": {"warmup_runs": 2, "max_tokens": 512},
        "generation_results": [
            {
                "prompt_text": "Hello world",
                "prompt_category": "test",
                "result": {
                    "text": "Hi there",
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "time_to_first_token": 0.1,
                    "total_time": 1.0,
                    "finish_reason": "stop",
                },
            }
        ],
        "metrics": {
            "ttft_mean": 0.1,
            "tokens_per_second_mean": 10.0,
            "total_tokens_generated": 10,
        },
        "model_metadata": {"family": "llama"},
        "hardware_summary": {"peak_vram_gb": 4.5},
    }


class TestStorage:
    def test_save_and_get_run(self, storage, sample_run_data):
        run_id = storage.save_run(**sample_run_data)
        assert run_id is not None
        assert run_id > 0

        run = storage.get_run(run_id)
        assert run is not None
        assert run["adapter_name"] == "ollama"
        assert run["model_id"] == "llama3:8b"
        assert run["profile"] == "quick"
        assert run["model_metadata"]["family"] == "llama"
        assert run["hardware_summary"]["peak_vram_gb"] == 4.5
        assert len(run["generation_results"]) == 1
        assert run["generation_results"][0]["result"]["text"] == "Hi there"
        assert run["metrics"]["ttft_mean"] == 0.1

    def test_get_nonexistent_run(self, storage):
        assert storage.get_run(9999) is None

    def test_list_runs(self, storage, sample_run_data):
        storage.save_run(**sample_run_data)
        sample_run_data["model_id"] = "mistral:7b"
        storage.save_run(**sample_run_data)

        runs = storage.list_runs()
        assert len(runs) == 2

    def test_list_runs_filter_model(self, storage, sample_run_data):
        storage.save_run(**sample_run_data)
        sample_run_data["model_id"] = "mistral:7b"
        storage.save_run(**sample_run_data)

        runs = storage.list_runs(model="mistral")
        assert len(runs) == 1
        assert runs[0]["model_id"] == "mistral:7b"

    def test_list_runs_filter_adapter(self, storage, sample_run_data):
        storage.save_run(**sample_run_data)

        runs = storage.list_runs(adapter="ollama")
        assert len(runs) == 1

        runs = storage.list_runs(adapter="vllm")
        assert len(runs) == 0

    def test_list_runs_limit(self, storage, sample_run_data):
        for i in range(5):
            sample_run_data["model_id"] = f"model:{i}"
            storage.save_run(**sample_run_data)

        runs = storage.list_runs(limit=3)
        assert len(runs) == 3

    def test_delete_run(self, storage, sample_run_data):
        run_id = storage.save_run(**sample_run_data)
        assert storage.delete_run(run_id) is True
        assert storage.get_run(run_id) is None

    def test_delete_nonexistent(self, storage):
        assert storage.delete_run(9999) is False

    def test_get_comparison(self, storage, sample_run_data):
        id1 = storage.save_run(**sample_run_data)
        sample_run_data["model_id"] = "mistral:7b"
        id2 = storage.save_run(**sample_run_data)

        runs = storage.get_comparison([id1, id2])
        assert len(runs) == 2
        models = {r["model_id"] for r in runs}
        assert "llama3:8b" in models
        assert "mistral:7b" in models

    def test_save_without_optional_fields(self, storage):
        run_id = storage.save_run(
            adapter_name="ollama",
            model_id="test",
            profile="quick",
            config={},
            generation_results=[],
            metrics={"ttft_mean": 0},
        )
        run = storage.get_run(run_id)
        assert run is not None
        assert run["model_metadata"] is None
        assert run["hardware_summary"] is None
