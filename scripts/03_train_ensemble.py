"""Stage 3: train the ensemble (best arch x 5 folds) and save members.

Reads results/best_per_model.json from stage 2; writes results/ensemble/.

Usage:
    python scripts/03_train_ensemble.py [--config ...] [--limit N]
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data.dataset import assert_no_leakage, load_benchmark, load_training_peptides
from src.data.embeddings import EmbeddingCache
from src.training.ensemble import build_ensemble
from src.utils.config import load_config, resolve_path
from src.utils.io import read_json
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

log = get_logger("stage3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))

    results_dir = Path(cfg["paths"]["results_dir"])
    best_per_model = read_json(results_dir / "best_per_model.json")

    peptides = load_training_peptides(
        resolve_path(cfg, "train_pos"), resolve_path(cfg, "train_neg")
    )
    benchmark = load_benchmark(resolve_path(cfg, "classification_benchmark"))
    peptides = assert_no_leakage(peptides, benchmark, drop=True)
    if args.limit:
        peptides = peptides[: args.limit]

    cache = EmbeddingCache(cfg["embeddings"]["cache"])
    ensemble = build_ensemble(cfg, cache, peptides, best_per_model)
    ensemble.save(results_dir / "ensemble")


if __name__ == "__main__":
    main()
