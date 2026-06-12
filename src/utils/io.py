"""Small JSON / pickle / dir helpers."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


def write_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def save_pickle(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)
