"""Stage 4: evaluate the ensemble on the full-length benchmark.

For each benchmark sequence: slide a window, score every residue, apply the
>0.5 / run>10 rule, and compare against ground-truth labels and core regions.
Writes results/benchmark_report.json and results/profiles.json.

Usage:
    python scripts/04_evaluate_benchmark.py [--config ...]
"""
import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from src.data.dataset import load_benchmark
from src.data.embeddings import EmbeddingCache, generate_embeddings
from src.evaluation.classify import classify_profile
from src.evaluation.metrics import region_iou, sequence_metrics
from src.training.ensemble import Ensemble
from src.utils.config import load_config, resolve_path
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

log = get_logger("stage4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))
    results_dir = Path(cfg["paths"]["results_dir"])

    benchmark = load_benchmark(resolve_path(cfg, "classification_benchmark"))
    cache = EmbeddingCache(cfg["embeddings"]["cache"])

    # Ensure full-length benchmark embeddings exist (windows slice from these).
    missing = cache.missing(b.sequence for b in benchmark)
    if missing:
        log.info("Generating embeddings for %d benchmark sequences", len(missing))
        generate_embeddings([b.sequence for b in benchmark], cfg)

    ensemble = Ensemble.load(results_dir / "ensemble")

    from src.evaluation.sliding import score_sequence

    thr = cfg["inference"]["residue_threshold"]
    min_run = cfg["inference"]["min_consecutive"]

    y_true, y_pred, profiles, ious = [], [], {}, []
    for b in benchmark:
        profile = score_sequence(b.sequence, ensemble, cache, cfg)
        label, run, pos_mask = classify_profile(profile, thr, min_run)
        y_true.append(b.label)
        y_pred.append(label)
        profiles[b.id] = {
            "label_true": b.label, "label_pred": label, "longest_run": int(run),
            "profile": [round(float(x), 4) for x in profile],
        }
        if b.label == 1 and b.core_regions:
            ious.append(region_iou(pos_mask, b.core_regions))

    seq = sequence_metrics(y_true, y_pred)
    report = {
        "sequence_level": seq,
        "n_sequences": len(benchmark),
        "mean_region_iou": float(np.nanmean(ious)) if ious else None,
        "inference": {"residue_threshold": thr, "min_consecutive": min_run,
                      "window_len": cfg["window_len"], "stride": cfg["inference"]["window_stride"]},
        "ensemble_members": len(ensemble.members),
    }
    write_json(report, results_dir / "benchmark_report.json")
    write_json(profiles, results_dir / "profiles.json")
    log.info("Benchmark: ACC=%.3f MCC=%.3f F1=%.3f (TP=%d FP=%d TN=%d FN=%d)",
             seq["accuracy"], seq["mcc"], seq["f1"], seq["tp"], seq["fp"], seq["tn"], seq["fn"])
    log.info("Report -> %s", results_dir / "benchmark_report.json")


if __name__ == "__main__":
    main()
