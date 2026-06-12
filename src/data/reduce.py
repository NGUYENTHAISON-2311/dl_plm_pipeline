"""Per-branch dimensionality reduction.

The concatenated pLM features are very wide (1024 + 1280) for a small dataset, so we
fit a reducer PER BRANCH on TRAIN-FOLD residues only and re-apply it to val/benchmark
to avoid leakage. PCA is the default; an autoencoder variant is provided behind the
``type: autoencoder`` config switch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..utils.io import load_pickle, save_pickle
from ..utils.logging import get_logger

log = get_logger(__name__)


class BranchReducer:
    """Wraps a single-branch reducer (PCA / autoencoder / identity)."""

    def __init__(self, kind: str = "pca", dim: int = 64, whiten: bool = False, seed: int = 42):
        self.kind = kind
        self.dim = dim
        self.whiten = whiten
        self.seed = seed
        self._model = None
        self._mean = None  # used by autoencoder standardisation

    def fit(self, residues: np.ndarray) -> "BranchReducer":
        """Fit on a (N_residues, D) matrix of train-fold residue embeddings."""
        if self.kind == "none":
            return self
        if self.kind == "pca":
            from sklearn.decomposition import PCA

            k = min(self.dim, residues.shape[1], residues.shape[0])
            self._model = PCA(n_components=k, whiten=self.whiten, random_state=self.seed)
            self._model.fit(residues)
            ev = float(np.sum(self._model.explained_variance_ratio_))
            log.info("PCA fit: %d -> %d dims (%.1f%% var)", residues.shape[1], k, 100 * ev)
        elif self.kind == "autoencoder":
            self._fit_autoencoder(residues)
        else:
            raise ValueError(f"Unknown reducer kind: {self.kind}")
        return self

    def transform(self, residues: np.ndarray) -> np.ndarray:
        """Reduce a (..., D) array along the last axis, preserving leading shape."""
        if self.kind == "none" or self._model is None:
            return residues
        flat = residues.reshape(-1, residues.shape[-1])
        if self.kind == "pca":
            red = self._model.transform(flat)
        else:  # autoencoder encoder
            red = self._encoder.predict((flat - self._mean), verbose=0)
        return red.reshape(*residues.shape[:-1], red.shape[-1])

    # -- autoencoder -------------------------------------------------------- #
    def _fit_autoencoder(self, residues: np.ndarray) -> None:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        self._mean = residues.mean(axis=0, keepdims=True)
        x = residues - self._mean
        d_in = x.shape[1]
        inp = layers.Input(shape=(d_in,))
        h = layers.Dense(256, activation="relu")(inp)
        z = layers.Dense(self.dim, activation="linear", name="code")(h)
        h2 = layers.Dense(256, activation="relu")(z)
        out = layers.Dense(d_in, activation="linear")(h2)
        ae = models.Model(inp, out)
        ae.compile(optimizer="adam", loss="mse")
        ae.fit(x, x, epochs=30, batch_size=256, verbose=0)
        self._model = ae
        self._encoder = models.Model(inp, z)
        log.info("Autoencoder fit: %d -> %d dims", d_in, self.dim)

    @property
    def out_dim(self) -> int:
        if self.kind == "none":
            raise ValueError("identity reducer has no fixed out_dim")
        return self.dim

    def save(self, path: str | Path) -> None:
        if self.kind == "autoencoder":
            # Keras models can't be pickled with the rest; save weights separately.
            self._encoder.save(str(Path(path).with_suffix(".keras")))
            save_pickle({"kind": self.kind, "dim": self.dim, "mean": self._mean}, path)
        else:
            save_pickle(self, path)

    @staticmethod
    def load(path: str | Path) -> "BranchReducer":
        obj = load_pickle(path)
        if isinstance(obj, BranchReducer):
            return obj
        # autoencoder case
        import tensorflow as tf

        r = BranchReducer(kind="autoencoder", dim=obj["dim"])
        r._mean = obj["mean"]
        r._encoder = tf.keras.models.load_model(str(Path(path).with_suffix(".keras")))
        return r


@dataclass
class DualReducer:
    """Holds the ProtT5 and ESM2 reducers together."""

    prott5: BranchReducer
    esm2: BranchReducer

    @classmethod
    def from_config(cls, cfg: dict, dim: Optional[int] = None) -> "DualReducer":
        rc = cfg["reducer"]
        d = dim if dim is not None else rc["dim"]
        seed = cfg.get("seed", 42)
        mk = lambda: BranchReducer(kind=rc["type"], dim=d, whiten=rc.get("whiten", False), seed=seed)
        return cls(prott5=mk(), esm2=mk())
