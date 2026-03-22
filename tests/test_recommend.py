"""Tests for the model recommendation engine."""

import pytest
from unittest.mock import patch, MagicMock

from local_inferbench.recommend import (
    ModelRecommendation,
    estimate_vram_gb,
    recommend_models,
    suggest_models_to_pull,
    _parse_param_size,
    _classify_model,
)


class TestParseParamSize:
    def test_billions(self):
        assert _parse_param_size("7B") == 7.0

    def test_billions_decimal(self):
        assert _parse_param_size("1.7B") == 1.7

    def test_millions(self):
        assert abs(_parse_param_size("500M") - 0.5) < 0.01

    def test_no_unit(self):
        assert _parse_param_size("7") == 7.0

    def test_invalid(self):
        assert _parse_param_size("unknown") == 0.0

    def test_whitespace(self):
        assert _parse_param_size("  8B  ") == 8.0


class TestEstimateVramGb:
    def test_7b_q4(self):
        vram = estimate_vram_gb(7.0, "Q4_K_M")
        assert 4.0 < vram < 7.0  # ~3.94 + 1.5 overhead

    def test_7b_f16(self):
        vram = estimate_vram_gb(7.0, "F16")
        assert 14.0 < vram < 17.0  # 14 + 1.5 overhead

    def test_70b_q4(self):
        vram = estimate_vram_gb(70.0, "Q4_K_M")
        assert 38.0 < vram < 45.0

    def test_small_model(self):
        vram = estimate_vram_gb(0.6, "Q4_K_M")
        assert 1.5 < vram < 3.0  # Mostly overhead

    def test_unknown_quant_defaults(self):
        vram = estimate_vram_gb(7.0, "UNKNOWN_QUANT")
        assert vram > 0  # Should use default Q4_K_M bytes


class TestClassifyModel:
    def test_optimal_with_headroom(self):
        tier, fits_vram, fits_ram, notes = _classify_model(4.0, 12.0, 32.0)
        assert tier == "optimal"
        assert fits_vram is True
        assert "headroom" in notes

    def test_optimal_tight(self):
        tier, fits_vram, _, notes = _classify_model(10.0, 12.0, 32.0)
        assert tier == "optimal"
        assert "Tight" in notes

    def test_feasible_partial_offload(self):
        tier, fits_vram, _, notes = _classify_model(15.0, 12.0, 32.0)
        assert tier == "feasible"
        assert fits_vram is False
        assert "offload" in notes.lower()

    def test_feasible_cpu_only(self):
        tier, _, fits_ram, notes = _classify_model(8.0, None, 32.0)
        assert tier == "feasible"
        assert fits_ram is True
        assert "CPU" in notes or "no GPU" in notes

    def test_too_large(self):
        tier, _, _, _ = _classify_model(100.0, 12.0, 32.0)
        assert tier == "too_large"

    def test_no_gpu_small_model(self):
        tier, fits_vram, fits_ram, _ = _classify_model(4.0, None, 16.0)
        assert tier == "feasible"
        assert fits_vram is False
        assert fits_ram is True


class TestRecommendModels:
    @patch("local_inferbench.recommend.get_available_models")
    @patch("local_inferbench.recommend.detect_hardware")
    def test_basic_recommendation(self, mock_hw, mock_models):
        mock_hw.return_value = {
            "gpus": [{"vram_total_gb": 12.0}],
            "ram_total_gb": 32.0,
        }
        mock_models.return_value = [
            {"name": "qwen3:0.6b", "parameter_size": "0.6B", "quantization_level": "Q4_K_M", "size_bytes": 0},
            {"name": "qwen3:8b", "parameter_size": "8B", "quantization_level": "Q4_K_M", "size_bytes": 0},
            {"name": "llama3:70b", "parameter_size": "70B", "quantization_level": "Q4_K_M", "size_bytes": 0},
        ]

        recs = recommend_models()
        assert len(recs) == 3

        # Small model should be optimal
        small = next(r for r in recs if "0.6b" in r.model_name)
        assert small.tier == "optimal"

        # 8B should be optimal for 12GB VRAM
        medium = next(r for r in recs if "8b" in r.model_name)
        assert medium.tier == "optimal"

        # 70B should be too large or feasible
        large = next(r for r in recs if "70b" in r.model_name)
        assert large.tier in ("feasible", "too_large")

    @patch("local_inferbench.recommend.get_available_models")
    @patch("local_inferbench.recommend.detect_hardware")
    def test_sorted_by_tier(self, mock_hw, mock_models):
        mock_hw.return_value = {"gpus": [{"vram_total_gb": 8.0}], "ram_total_gb": 16.0}
        mock_models.return_value = [
            {"name": "big:70b", "parameter_size": "70B", "quantization_level": "Q4_K_M", "size_bytes": 0},
            {"name": "small:1b", "parameter_size": "1B", "quantization_level": "Q4_K_M", "size_bytes": 0},
        ]

        recs = recommend_models()
        assert recs[0].model_name == "small:1b"  # Optimal first

    @patch("local_inferbench.recommend.get_available_models")
    def test_no_models_returns_empty(self, mock_models):
        mock_models.return_value = []
        recs = recommend_models(hardware={"gpus": [], "ram_total_gb": 8.0})
        assert recs == []


class TestSuggestModelsToPull:
    def test_suggests_for_small_gpu(self):
        hw = {"gpus": [{"vram_total_gb": 4.0}], "ram_total_gb": 16.0}
        suggestions = suggest_models_to_pull(hardware=hw, installed_names=set())
        assert len(suggestions) > 0
        # Should only suggest models that fit
        for s in suggestions:
            assert s.tier in ("optimal", "feasible")

    def test_skips_installed(self):
        hw = {"gpus": [{"vram_total_gb": 24.0}], "ram_total_gb": 64.0}
        installed = {"qwen3:0.6b", "llama3.2:3b"}
        suggestions = suggest_models_to_pull(hardware=hw, installed_names=installed)
        names = {s.model_name for s in suggestions}
        assert "qwen3:0.6b" not in names
        assert "llama3.2:3b" not in names

    def test_no_gpu(self):
        hw = {"gpus": [], "ram_total_gb": 16.0}
        suggestions = suggest_models_to_pull(hardware=hw, installed_names=set())
        for s in suggestions:
            assert s.fits_in_ram is True
            assert s.fits_in_vram is False
