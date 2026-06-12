"""Config loading and lightweight dotted-access helpers."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def load_config(path: str | Path = "config/default.yaml") -> Dict[str, Any]:
    """Load the main config, resolving paths relative to the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return load_yaml(path)


def load_model_config(name: str) -> Dict[str, Any]:
    """Load a per-model grid config from config/models/<name>.yaml."""
    return load_yaml(PROJECT_ROOT / "config" / "models" / f"{name}.yaml")


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve a path stored under cfg['paths'][key] against the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
