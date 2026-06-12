"""Stage 2: grid-search hyperparameters (+ reducer dim) per architecture via 5-fold CV.

Writes results/gridsearch.csv, results/best_per_model.json and results/best/<model>.json.

Usage:
    python scripts/02_run_gridsearch.py [--config ...] [--models fnn cnn] [--limit N]
"""
import argparse

import _bootstrap  # noqa: F401

from src.data.dataset import assert_no_leakage, load_benchmark, load_training_peptides
from src.data.embeddings import EmbeddingCache
from src.training.grid_search import run_gridsearch
from src.utils.config import load_config, resolve_path
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

log = get_logger("stage2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--models", nargs="*", default=None, help="override gridsearch.models")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))
    if args.models:
        cfg["gridsearch"]["models"] = args.models

    peptides = load_training_peptides(
        resolve_path(cfg, "train_pos"), resolve_path(cfg, "train_neg")
    )
    benchmark = load_benchmark(resolve_path(cfg, "classification_benchmark"))
    peptides = assert_no_leakage(peptides, benchmark, drop=True)
    if args.limit:
        peptides = peptides[: args.limit]

    cache = EmbeddingCache(cfg["embeddings"]["cache"])
    missing = cache.missing(p.sequence for p in peptides)
    if missing:
        raise SystemExit(f"{len(missing)} peptides missing embeddings; run stage 1 first.")

    run_gridsearch(cfg, cache, peptides)


if __name__ == "__main__":
    main()
