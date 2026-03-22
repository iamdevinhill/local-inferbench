"""Prompt profile registry — discovers and loads YAML prompt profiles."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class Prompt(BaseModel):
    """A single benchmark prompt."""

    model_config = ConfigDict(strict=True)

    category: str
    text: str


class PromptProfile(BaseModel):
    """A collection of prompts for benchmarking."""

    model_config = ConfigDict(strict=True)

    name: str
    description: str
    prompts: list[Prompt]


_PROMPTS_DIR = Path(__file__).parent


def _discover_profiles() -> dict[str, Path]:
    """Find all YAML files in the prompts directory."""
    profiles: dict[str, Path] = {}
    for f in _PROMPTS_DIR.iterdir():
        if f.suffix in (".yaml", ".yml") and f.is_file():
            profiles[f.stem] = f
    return profiles


def list_profiles() -> list[dict[str, str]]:
    """Return a list of available profile names and descriptions."""
    result = []
    for name, path in sorted(_discover_profiles().items()):
        with open(path) as f:
            data = yaml.safe_load(f)
        result.append({
            "name": data.get("name", name),
            "description": data.get("description", ""),
            "prompt_count": str(len(data.get("prompts", []))),
        })
    return result


def load_profile(name_or_path: str) -> PromptProfile:
    """Load a prompt profile by name or file path.

    If name_or_path is a path to an existing file, load it directly.
    Otherwise, look it up in the built-in profiles directory.
    """
    if os.path.isfile(name_or_path):
        path = Path(name_or_path)
    else:
        profiles = _discover_profiles()
        if name_or_path not in profiles:
            available = ", ".join(sorted(profiles.keys()))
            raise ValueError(
                f"Unknown profile '{name_or_path}'. Available: {available}"
            )
        path = profiles[name_or_path]

    with open(path) as f:
        data = yaml.safe_load(f)

    return PromptProfile(**data)
