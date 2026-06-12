"""Stage 4: evaluate the ensemble on the full-length benchmark.

The PLMs run inference DIRECTLY on each sliding window (window_len residues) at score
time -- no cached benchmark embeddings. For each sequence we slide a window, embed every
window with ProtT5 + ESM2, score residues with the ensemble, aggregate, then apply the
>0.5 / run>10 rule. Writes results/benchmark_report.json, profiles.json, windows.json.

Usage:
    python scripts/04_evaluate_benchmark.py [--config ...] [--device auto|cpu|cuda]
"""
import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from src.data.dataset import load_benchmark
from src.data.embeddings import PLMEmbedder
from src.evaluation.classify import classify_profile
from src.evaluation.metrics import region_iou, sequence_metrics
from src.evaluation.sliding import score_sequence_live
from src.training.ensemble import Ensemble
from src.utils.config import load_config, resolve_path
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

log = get_logger("stage4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--device", default=None, help="auto|cpu|cuda for the PLMs (overrides config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))
    results_dir = Path(cfg["paths"]["results_dir"])

    benchmark = load_benchmark(resolve_path(cfg, "classification_benchmark"))
    ensemble = Ensemble.load(results_dir / "ensemble")

    # Live PLM inference on each window (no cache).
    device = args.device or cfg["embeddings"].get("device", "auto")
    embedder = PLMEmbedder(cfg, device=device)

    thr = cfg["inference"]["residue_threshold"]
    min_run = cfg["inference"]["min_consecutive"]

    y_true, y_pred, profiles, windows_out, ious = [], [], {}, {}, []
    for b in benchmark:
        profile, windows = score_sequence_live(b.sequence, embedder, ensemble, cfg)
        label, run, pos_mask = classify_profile(profile, thr, min_run)
        y_true.append(b.label)
        y_pred.append(label)
        profiles[b.id] = {
            "label_true": b.label, "label_pred": label, "longest_run": int(run),
            "profile": [round(float(x), 4) for x in profile],
        }
        windows_out[b.id] = windows
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
    write_json(windows_out, results_dir / "windows.json")
    log.info("Benchmark: ACC=%.3f MCC=%.3f F1=%.3f (TP=%d FP=%d TN=%d FN=%d)",
             seq["accuracy"], seq["mcc"], seq["f1"], seq["tp"], seq["fp"], seq["tn"], seq["fn"])
    log.info("Report -> %s", results_dir / "benchmark_report.json")


if __name__ == "__main__":
    main()
