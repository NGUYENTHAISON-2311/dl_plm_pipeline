"""Residue-level and sequence-level metric helpers."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _flatten_masked(y: np.ndarray, p: np.ndarray, mask: np.ndarray):
    m = mask.astype(bool).reshape(-1)
    return y.reshape(-1)[m], p.reshape(-1)[m]


def residue_metrics(y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """AUROC / AUPRC / MCC / F1 / precision / recall over real (unmasked) residues."""
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    yt, ps = _flatten_masked(y_true, y_score, mask)
    yp = (ps > threshold).astype(int)
    out = {"n_residues": int(yt.size), "n_positive": int(yt.sum())}
    # Guard degenerate single-class cases.
    out["auroc"] = float(roc_auc_score(yt, ps)) if len(np.unique(yt)) > 1 else float("nan")
    out["auprc"] = float(average_precision_score(yt, ps)) if yt.sum() > 0 else float("nan")
    out["mcc"] = float(matthews_corrcoef(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
    out["f1"] = float(f1_score(yt, yp, zero_division=0))
    out["precision"] = float(precision_score(yt, yp, zero_division=0))
    out["recall"] = float(recall_score(yt, yp, zero_division=0))
    return out


def sequence_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """ACC / MCC / F1 + confusion matrix entries for the 14-vs-14 benchmark call."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
    )

    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "mcc": float(matthews_corrcoef(yt, yp)) if len(np.unique(yt)) > 1 else float("nan"),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def region_iou(pred_mask: np.ndarray, true_regions: Sequence[dict]) -> float:
    """IoU between predicted positive residues and annotated core regions.

    ``true_regions`` are dicts with ``start``/``end`` (1-based, inclusive) as found in
    the benchmark ``matched_core_regions``.
    """
    if not true_regions:
        return float("nan")
    truth = np.zeros_like(pred_mask, dtype=bool)
    for r in true_regions:
        s = max(int(r.get("start", 1)) - 1, 0)
        e = min(int(r.get("end", 0)), pred_mask.size)
        truth[s:e] = True
    pred = pred_mask.astype(bool)
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return float(inter / union) if union else float("nan")
