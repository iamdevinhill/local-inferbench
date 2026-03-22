"""Automatic model recommendation based on detected hardware.

Estimates VRAM requirements for models and classifies them into tiers
(optimal, feasible, too_large) based on available GPU VRAM and system RAM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

from local_inferbench.hardware import detect_hardware

logger = logging.getLogger(__name__)

# Bytes per parameter by quantization type
_BYTES_PER_PARAM: dict[str, float] = {
    "Q2_K": 0.3125,
    "Q3_K_S": 0.375,
    "Q3_K_M": 0.40,
    "Q3_K_L": 0.4375,
    "Q4_0": 0.5,
    "Q4_K_S": 0.5,
    "Q4_K_M": 0.5625,
    "Q4_1": 0.5625,
    "Q5_0": 0.625,
    "Q5_K_S": 0.625,
    "Q5_K_M": 0.6875,
    "Q5_1": 0.6875,
    "Q6_K": 0.75,
    "Q8_0": 1.0,
    "F16": 2.0,
    "F32": 4.0,
}

# KV cache + overhead estimate in GB
_OVERHEAD_GB = 1.5

# Curated list of well-known models for suggestions
_SUGGESTED_MODELS: list[dict[str, str | float]] = [
    {"name": "qwen3:0.6b", "params_b": 0.6, "quant": "Q4_K_M", "description": "Tiny, fast, good for testing"},
    {"name": "qwen3:1.7b", "params_b": 1.7, "quant": "Q4_K_M", "description": "Small, solid reasoning"},
    {"name": "llama3.2:1b", "params_b": 1.0, "quant": "Q4_K_M", "description": "Lightweight Meta model"},
    {"name": "llama3.2:3b", "params_b": 3.0, "quant": "Q4_K_M", "description": "Good balance of speed/quality"},
    {"name": "phi4-mini:3.8b", "params_b": 3.8, "quant": "Q4_K_M", "description": "Microsoft small model"},
    {"name": "mistral:7b", "params_b": 7.0, "quant": "Q4_K_M", "description": "Strong 7B all-rounder"},
    {"name": "llama3.1:8b", "params_b": 8.0, "quant": "Q4_K_M", "description": "Meta's workhorse model"},
    {"name": "qwen3:8b", "params_b": 8.0, "quant": "Q4_K_M", "description": "Strong reasoning and code"},
    {"name": "gemma2:9b", "params_b": 9.0, "quant": "Q4_K_M", "description": "Google's efficient model"},
    {"name": "codellama:13b", "params_b": 13.0, "quant": "Q4_K_M", "description": "Code-specialized 13B"},
    {"name": "llama3.1:70b", "params_b": 70.0, "quant": "Q4_K_M", "description": "Large, needs ~42GB VRAM"},
    {"name": "qwen3:30b", "params_b": 30.0, "quant": "Q4_K_M", "description": "Large reasoning model"},
]


@dataclass
class ModelRecommendation:
    """Recommendation for a single model based on hardware capabilities."""

    model_name: str
    parameter_size: str
    quantization: str
    estimated_vram_gb: float
    fits_in_vram: bool
    fits_in_ram: bool
    recommended: bool
    tier: str  # "optimal", "feasible", "too_large"
    notes: str


def _parse_param_size(size_str: str) -> float:
    """Parse parameter size string like '7B', '1.7B', '500M' to billions."""
    size_str = size_str.strip().upper()
    match = re.match(r"([\d.]+)\s*([BMK]?)", size_str)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "M":
        return value / 1000.0
    elif unit == "K":
        return value / 1_000_000.0
    return value  # Already in billions (or no unit)


def estimate_vram_gb(params_billions: float, quantization: str) -> float:
    """Estimate VRAM needed in GB for a model with given size and quantization."""
    # Normalize quantization string
    quant_upper = quantization.upper().replace("-", "_")

    # Find the best matching quantization
    bytes_per = _BYTES_PER_PARAM.get(quant_upper)
    if bytes_per is None:
        # Try partial match
        for key, val in _BYTES_PER_PARAM.items():
            if key in quant_upper or quant_upper in key:
                bytes_per = val
                break
    if bytes_per is None:
        bytes_per = 0.5625  # Default to Q4_K_M

    model_size_gb = params_billions * bytes_per
    return round(model_size_gb + _OVERHEAD_GB, 2)


def _classify_model(
    estimated_vram: float,
    gpu_vram_gb: float | None,
    ram_gb: float,
) -> tuple[str, bool, bool, str]:
    """Classify a model into a tier. Returns (tier, fits_vram, fits_ram, notes)."""
    fits_ram = estimated_vram < ram_gb * 0.8  # Leave 20% headroom
    fits_vram = gpu_vram_gb is not None and estimated_vram < gpu_vram_gb * 0.9  # 10% headroom
    has_gpu = gpu_vram_gb is not None and gpu_vram_gb > 0

    if fits_vram:
        headroom = gpu_vram_gb - estimated_vram  # type: ignore[operator]
        if headroom > 2.0:
            return "optimal", True, fits_ram, f"{headroom:.1f} GB VRAM headroom"
        else:
            return "optimal", True, fits_ram, f"Tight fit — {headroom:.1f} GB VRAM headroom"

    if has_gpu and estimated_vram < gpu_vram_gb * 1.5:  # type: ignore[operator]
        return "feasible", False, fits_ram, "Will offload some layers to CPU"

    if fits_ram:
        if has_gpu:
            return "feasible", False, True, "Too large for GPU — CPU-only inference"
        else:
            return "feasible", False, True, "CPU-only inference (no GPU detected)"

    return "too_large", False, False, "Exceeds both GPU VRAM and available RAM"


def get_available_models(base_url: str = "http://localhost:11434") -> list[dict]:
    """Fetch locally installed models from Ollama with their metadata."""
    models = []
    try:
        client = httpx.Client(base_url=base_url, timeout=30.0)

        # List installed models
        resp = client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()

        for model_info in data.get("models", []):
            name = model_info.get("name", "")
            details = model_info.get("details", {})

            # Get more details via /api/show
            try:
                show_resp = client.post("/api/show", json={"name": name})
                show_resp.raise_for_status()
                show_data = show_resp.json()
                show_details = show_data.get("details", {})
            except Exception:
                show_details = details

            models.append({
                "name": name,
                "parameter_size": show_details.get("parameter_size", "unknown"),
                "quantization_level": show_details.get("quantization_level", "unknown"),
                "family": show_details.get("family", "unknown"),
                "size_bytes": model_info.get("size", 0),
            })

        client.close()
    except Exception as e:
        logger.warning("Failed to fetch models from Ollama: %s", e)

    return models


def recommend_models(
    base_url: str = "http://localhost:11434",
    hardware: dict | None = None,
) -> list[ModelRecommendation]:
    """Recommend installed models based on hardware capabilities."""
    if hardware is None:
        hardware = detect_hardware()

    gpus = hardware.get("gpus", [])
    gpu_vram_gb: float | None = None
    if gpus:
        gpu_vram_gb = max(g.get("vram_total_gb", 0) for g in gpus)
    ram_gb = float(hardware.get("ram_total_gb", 0))

    models = get_available_models(base_url)
    recommendations = []

    for model in models:
        param_str = model["parameter_size"]
        quant = model["quantization_level"]

        if param_str == "unknown":
            # Estimate from file size if available
            size_bytes = model.get("size_bytes", 0)
            if size_bytes > 0:
                params_b = size_bytes / (0.5625 * 1e9)  # Rough estimate assuming Q4_K_M
            else:
                params_b = 0.0
        else:
            params_b = _parse_param_size(param_str)

        estimated = estimate_vram_gb(params_b, quant if quant != "unknown" else "Q4_K_M")
        tier, fits_vram, fits_ram, notes = _classify_model(estimated, gpu_vram_gb, ram_gb)

        recommendations.append(ModelRecommendation(
            model_name=model["name"],
            parameter_size=param_str,
            quantization=quant,
            estimated_vram_gb=estimated,
            fits_in_vram=fits_vram,
            fits_in_ram=fits_ram,
            recommended=tier in ("optimal", "feasible"),
            tier=tier,
            notes=notes,
        ))

    # Sort: optimal first, then feasible, then too_large; within tier by VRAM ascending
    tier_order = {"optimal": 0, "feasible": 1, "too_large": 2}
    recommendations.sort(key=lambda r: (tier_order.get(r.tier, 3), r.estimated_vram_gb))

    return recommendations


def suggest_models_to_pull(
    hardware: dict | None = None,
    installed_names: set[str] | None = None,
) -> list[ModelRecommendation]:
    """Suggest well-known models to download based on hardware."""
    if hardware is None:
        hardware = detect_hardware()
    if installed_names is None:
        installed_names = set()

    gpus = hardware.get("gpus", [])
    gpu_vram_gb: float | None = None
    if gpus:
        gpu_vram_gb = max(g.get("vram_total_gb", 0) for g in gpus)
    ram_gb = float(hardware.get("ram_total_gb", 0))

    suggestions = []
    for model in _SUGGESTED_MODELS:
        name = str(model["name"])

        # Skip already installed
        if any(name.split(":")[0] in inst for inst in installed_names):
            continue

        params_b = float(model["params_b"])
        quant = str(model["quant"])
        estimated = estimate_vram_gb(params_b, quant)
        tier, fits_vram, fits_ram, notes = _classify_model(estimated, gpu_vram_gb, ram_gb)

        if tier in ("optimal", "feasible"):
            suggestions.append(ModelRecommendation(
                model_name=name,
                parameter_size=f"{params_b}B",
                quantization=quant,
                estimated_vram_gb=estimated,
                fits_in_vram=fits_vram,
                fits_in_ram=fits_ram,
                recommended=True,
                tier=tier,
                notes=f"{model['description']} — {notes}",
            ))

    tier_order = {"optimal": 0, "feasible": 1}
    suggestions.sort(key=lambda r: (tier_order.get(r.tier, 2), r.estimated_vram_gb))
    return suggestions
