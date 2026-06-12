"""Config-driven grid search over hyperparameters and reducer dim, scored by CV."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from ..data.dataset import Peptide
from ..data.embeddings import EmbeddingCache
from ..models.registry import encoder_for, grid_for
from ..utils.io import ensure_dir, write_json
from ..utils.logging import get_logger
from .cv import make_folds, subset

log = get_logger(__name__)


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of all grid axes -> list of hyperparameter dicts."""
    keys = list(grid.keys())
    combos = itertools.product(*[grid[k] for k in keys])
    return [dict(zip(keys, vals)) for vals in combos]


def default_best_per_model(cfg: dict) -> Dict[str, Any]:
    """Skip grid search: use each model's first grid combo as its 'best' config.

    Lets the notebook/ensemble run without a (slow) search. Replace with the output of
    ``run_gridsearch`` for tuned hyperparameters.
    """
    out: Dict[str, Any] = {}
    for model_name in cfg["gridsearch"]["models"]:
        out[model_name] = {
            "encoder": encoder_for(model_name),
            "best_hp": expand_grid(grid_for(model_name))[0],
            "best_score": None,
        }
    return out


def search_model(
    model_name: str,
    cfg: dict,
    cache: EmbeddingCache,
    peptides: Sequence[Peptide],
    folds: List,
) -> Dict[str, Any]:
    """Grid-search one architecture; return the best hyperparameters + CV score."""
    from .train import train_fold  # lazy: avoids importing TF for CV/grid utilities

    encoder = encoder_for(model_name)
    combos = expand_grid(grid_for(model_name))
    metric = cfg["gridsearch"]["metric"].replace("residue_", "")
    log.info("Grid search '%s': %d combos x %d folds", model_name, len(combos), len(folds))

    rows: List[Dict[str, Any]] = []
    best = {"score": -np.inf, "hp": None}

    for ci, hp in enumerate(combos):
        fold_scores = []
        for fi, (tr_idx, va_idx) in enumerate(folds):
            res = train_fold(
                encoder, hp, cfg, cache,
                subset(peptides, tr_idx), subset(peptides, va_idx),
            )
            fold_scores.append(res.val_metrics[metric])
        mean_s, std_s = float(np.mean(fold_scores)), float(np.std(fold_scores))
        rows.append({"model": model_name, "combo": ci, **hp,
                     f"cv_{metric}_mean": mean_s, f"cv_{metric}_std": std_s})
        log.info("  combo %d/%d  cv_%s=%.3f±%.3f", ci + 1, len(combos), metric, mean_s, std_s)
        if mean_s > best["score"]:
            best = {"score": mean_s, "hp": hp}

    return {"model": model_name, "encoder": encoder, "best_hp": best["hp"],
            "best_score": best["score"], "rows": rows}


def run_gridsearch(
    cfg: dict, cache: EmbeddingCache, peptides: Sequence[Peptide]
) -> Dict[str, Any]:
    """Grid-search every configured model; persist results and best configs."""
    folds = make_folds(peptides, cfg)
    results_dir = ensure_dir(Path(cfg["paths"]["results_dir"]))
    best_dir = ensure_dir(results_dir / "best")

    all_rows: List[Dict[str, Any]] = []
    best_per_model: Dict[str, Any] = {}
    for model_name in cfg["gridsearch"]["models"]:
        out = search_model(model_name, cfg, cache, peptides, folds)
        all_rows.extend(out["rows"])
        best_per_model[model_name] = {
            "encoder": out["encoder"], "best_hp": out["best_hp"], "best_score": out["best_score"],
        }
        write_json(best_per_model[model_name], best_dir / f"{model_name}.json")

    pd.DataFrame(all_rows).to_csv(results_dir / "gridsearch.csv", index=False)
    write_json(best_per_model, results_dir / "best_per_model.json")
    log.info("Grid search complete -> %s", results_dir / "gridsearch.csv")
    return best_per_model
