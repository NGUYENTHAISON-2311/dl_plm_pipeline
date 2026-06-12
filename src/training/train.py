"""Fit a single two-branch model on one fold and report validation metrics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..data.dataset import Peptide
from ..data.embeddings import EmbeddingCache
from ..data.features import (
    build_tensors,
    fit_reducers,
    residue_class_weights,
    sample_weights_from_mask,
)
from ..data.reduce import DualReducer
from ..evaluation.metrics import residue_metrics
from ..models.builders import build_model
from ..utils.distribute import build_scope
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FoldResult:
    model: Any
    dual: DualReducer
    history: Dict[str, List[float]]
    val_metrics: Dict[str, float]
    val_scores: np.ndarray          # (Nval, L)
    val_labels: np.ndarray          # (Nval, L)
    val_mask: np.ndarray            # (Nval, L)


def _callbacks(cfg: dict, ckpt_path: Optional[str]):
    from tensorflow.keras import callbacks as kc

    t = cfg["train"]
    cbs = [
        kc.EarlyStopping(monitor=t["monitor"], mode="max",
                         patience=t["early_stopping_patience"], restore_best_weights=True),
        kc.ReduceLROnPlateau(monitor=t["monitor"], mode="max",
                             patience=t["reduce_lr_patience"], factor=0.5, min_lr=1e-6),
    ]
    if ckpt_path:
        cbs.append(kc.ModelCheckpoint(ckpt_path, monitor=t["monitor"], mode="max",
                                      save_best_only=True, save_weights_only=True))
    return cbs


def train_fold(
    encoder: str,
    hp: Dict[str, Any],
    cfg: dict,
    cache: EmbeddingCache,
    train_peptides: Sequence[Peptide],
    val_peptides: Sequence[Peptide],
    ckpt_path: Optional[str] = None,
) -> FoldResult:
    """Fit reducers on train peptides, build tensors, train, and score validation."""
    window_len = cfg["window_len"]
    reducer_dim = hp.get("reducer_dim", cfg["reducer"]["dim"])

    dual = fit_reducers(train_peptides, cache, cfg, dim=reducer_dim)
    Xtr_p, Xtr_e, Ytr, Mtr = build_tensors(train_peptides, cache, dual, window_len)
    Xva_p, Xva_e, Yva, Mva = build_tensors(val_peptides, cache, dual, window_len)

    imbalance = cfg.get("imbalance", {})
    cw = residue_class_weights(Ytr, Mtr) if imbalance.get("strategy") == "class_weight" else {0: 1.0, 1: 1.0}
    sw_tr = sample_weights_from_mask(Ytr, Mtr, cw)
    sw_va = Mva.astype(np.float32)  # weight val metrics by real residues only

    # Build/compile inside the distribution scope when train.distribute=mirrored.
    with build_scope(cfg):
        model = build_model(
            encoder=encoder, hp=hp, window_len=window_len,
            dim_prott5=dual.prott5.out_dim, dim_esm2=dual.esm2.out_dim,
            learning_rate=hp.get("learning_rate", cfg["train"]["learning_rate"]),
            imbalance=imbalance,
        )

    history = model.fit(
        {"prott5_in": Xtr_p, "esm2_in": Xtr_e}, Ytr,
        sample_weight=sw_tr,
        validation_data=({"prott5_in": Xva_p, "esm2_in": Xva_e}, Yva, sw_va),
        epochs=cfg["train"]["epochs"],
        batch_size=hp.get("batch_size", cfg["train"]["batch_size"]),
        callbacks=_callbacks(cfg, ckpt_path),
        verbose=0,
    )

    val_scores = model.predict({"prott5_in": Xva_p, "esm2_in": Xva_e}, verbose=0)
    val_metrics = residue_metrics(Yva, val_scores, Mva)
    log.info("[%s] fold val: AUPRC=%.3f AUROC=%.3f MCC=%.3f",
             encoder, val_metrics["auprc"], val_metrics["auroc"], val_metrics["mcc"])

    return FoldResult(
        model=model, dual=dual, history=history.history, val_metrics=val_metrics,
        val_scores=val_scores, val_labels=Yva, val_mask=Mva,
    )
