"""Train the ensemble (best arch x 5 folds) and average per-residue probabilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from ..data.dataset import Peptide
from ..data.embeddings import EmbeddingCache
from ..data.reduce import DualReducer
from ..utils.io import ensure_dir, write_json
from ..utils.logging import get_logger
from .cv import make_folds, subset
from .train import train_fold

log = get_logger(__name__)


@dataclass
class EnsembleMember:
    model: Any
    dual: DualReducer
    name: str


@dataclass
class Ensemble:
    """A bag of trained members whose per-residue sigmoids are averaged."""

    members: List[EnsembleMember] = field(default_factory=list)
    window_len: int = 20

    def predict_peptides(self, peptides: Sequence[Peptide], cache: EmbeddingCache) -> np.ndarray:
        """Mean per-residue score (N, L) for a list of peptides."""
        from ..data.features import build_tensors

        preds = []
        for m in self.members:
            Xp, Xe, _, _ = build_tensors(peptides, cache, m.dual, self.window_len)
            preds.append(m.model.predict({"prott5_in": Xp, "esm2_in": Xe}, verbose=0))
        return np.mean(preds, axis=0)

    def predict_window(self, emb: Dict[str, np.ndarray]) -> np.ndarray:
        """Mean per-residue score for one window given its raw embeddings dict.

        ``emb`` holds ``prott5`` (L,1024) and ``esm2`` (L,1280) for a single window of
        length <= window_len. Used by sliding-window inference.
        """
        L = emb["prott5"].shape[0]
        preds = []
        for m in self.members:
            xp = np.zeros((1, self.window_len, m.dual.prott5.out_dim), dtype=np.float32)
            xe = np.zeros((1, self.window_len, m.dual.esm2.out_dim), dtype=np.float32)
            xp[0, :L] = m.dual.prott5.transform(emb["prott5"])[:L]
            xe[0, :L] = m.dual.esm2.transform(emb["esm2"])[:L]
            preds.append(m.model.predict({"prott5_in": xp, "esm2_in": xe}, verbose=0)[0, :L])
        return np.mean(preds, axis=0)

    def save(self, out_dir: str | Path) -> None:
        out_dir = ensure_dir(out_dir)
        manifest = {"window_len": self.window_len, "members": []}
        for i, m in enumerate(self.members):
            m.model.save(str(out_dir / f"{m.name}.keras"))
            m.dual.prott5.save(out_dir / f"{m.name}_reducer_p.pkl")
            m.dual.esm2.save(out_dir / f"{m.name}_reducer_e.pkl")
            manifest["members"].append({"name": m.name})
        write_json(manifest, out_dir / "manifest.json")
        log.info("Saved ensemble (%d members) -> %s", len(self.members), out_dir)

    @classmethod
    def load(cls, out_dir: str | Path) -> "Ensemble":
        import tensorflow as tf

        from ..data.reduce import BranchReducer
        from ..models.builders import focal_loss, masked_bce
        from ..utils.io import read_json

        out_dir = Path(out_dir)
        manifest = read_json(out_dir / "manifest.json")
        custom = {"masked_bce": masked_bce, "loss": focal_loss()}
        ens = cls(window_len=manifest["window_len"])
        for entry in manifest["members"]:
            name = entry["name"]
            model = tf.keras.models.load_model(
                str(out_dir / f"{name}.keras"), custom_objects=custom, compile=False
            )
            dual = DualReducer(
                prott5=BranchReducer.load(out_dir / f"{name}_reducer_p.pkl"),
                esm2=BranchReducer.load(out_dir / f"{name}_reducer_e.pkl"),
            )
            ens.members.append(EnsembleMember(model, dual, name))
        log.info("Loaded ensemble (%d members) from %s", len(ens.members), out_dir)
        return ens


def _selected_models(cfg: dict, best_per_model: Dict[str, Any]) -> List[str]:
    if cfg["ensemble"].get("members") == "all":
        return list(best_per_model)
    return [m for m in cfg["gridsearch"]["models"] if m in best_per_model]


def build_ensemble(
    cfg: dict,
    cache: EmbeddingCache,
    peptides: Sequence[Peptide],
    best_per_model: Dict[str, Any],
) -> Ensemble:
    """Train every (architecture x fold) member declared by the config."""
    folds = make_folds(peptides, cfg)
    ens = Ensemble(window_len=cfg["window_len"])
    for model_name in _selected_models(cfg, best_per_model):
        spec = best_per_model[model_name]
        for fi, (tr_idx, va_idx) in enumerate(folds):
            res = train_fold(
                spec["encoder"], spec["best_hp"], cfg, cache,
                subset(peptides, tr_idx), subset(peptides, va_idx),
            )
            ens.members.append(EnsembleMember(res.model, res.dual, f"{model_name}_fold{fi}"))
            log.info("Ensemble member %s trained", f"{model_name}_fold{fi}")
    return ens


def build_ensemble_cv(
    cfg: dict,
    cache: EmbeddingCache,
    peptides: Sequence[Peptide],
    best_per_model: Dict[str, Any],
) -> tuple["Ensemble", Dict[str, Any], List[Dict[str, Any]]]:
    """Train the ensemble *and* collect honest out-of-fold (OOF) predictions.

    For each architecture, one member is trained per fold on that fold's train split.
    A peptide's OOF ensemble score is the mean, over architectures, of the prediction
    made by the member that held that peptide out (so no member ever scores a peptide
    it trained on). Returns ``(ensemble, oof, per_model_fold_metrics)`` where ``oof``
    holds per-residue ``scores``/``labels``/``mask`` (N, L) aligned to ``peptides``.
    """
    folds = make_folds(peptides, cfg)
    ens = Ensemble(window_len=cfg["window_len"])
    n, L = len(peptides), cfg["window_len"]
    oof_scores = np.zeros((n, L), dtype=np.float64)
    oof_labels = np.zeros((n, L), dtype=np.float32)
    oof_mask = np.zeros((n, L), dtype=np.float32)
    oof_count = np.zeros(n, dtype=np.float64)
    per_model_fold: List[Dict[str, Any]] = []

    for model_name in _selected_models(cfg, best_per_model):
        spec = best_per_model[model_name]
        for fi, (tr_idx, va_idx) in enumerate(folds):
            res = train_fold(
                spec["encoder"], spec["best_hp"], cfg, cache,
                subset(peptides, tr_idx), subset(peptides, va_idx),
            )
            ens.members.append(EnsembleMember(res.model, res.dual, f"{model_name}_fold{fi}"))
            oof_scores[va_idx] += res.val_scores
            oof_labels[va_idx] = res.val_labels
            oof_mask[va_idx] = res.val_mask
            oof_count[va_idx] += 1
            per_model_fold.append({"model": model_name, "fold": fi, **res.val_metrics})
            log.info("CV member %s trained", f"{model_name}_fold{fi}")

    oof_count[oof_count == 0] = 1.0
    oof_scores /= oof_count[:, None]
    oof = {
        "scores": oof_scores.astype(np.float32),
        "labels": oof_labels,
        "mask": oof_mask,
        "ids": [p.id for p in peptides],
    }
    return ens, oof, per_model_fold
