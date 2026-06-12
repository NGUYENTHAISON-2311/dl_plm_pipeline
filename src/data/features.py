"""Assemble model-ready tensors from cached per-residue embeddings.

Produces, for a list of peptides:
  - X_prott5 : (N, L, d_p)   reduced ProtT5 features, zero-padded to window L
  - X_esm2   : (N, L, d_e)   reduced ESM2 features, zero-padded to window L
  - Y        : (N, L)        per-residue labels (0/1), padding rows = 0
  - mask     : (N, L)        1 for real residues, 0 for padding (sample weights)

Reducers are fit on the *training* peptides of a fold only, then reused for val.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .dataset import Peptide
from .embeddings import EmbeddingCache
from .reduce import BranchReducer, DualReducer


def _stack_residues(peptides: Sequence[Peptide], cache: EmbeddingCache) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate all residues across peptides into (Ntot, d_p) / (Ntot, d_e)."""
    p_list, e_list = [], []
    for pep in peptides:
        emb = cache.get(pep.sequence)
        p_list.append(emb["prott5"])
        e_list.append(emb["esm2"])
    return np.concatenate(p_list, axis=0), np.concatenate(e_list, axis=0)


def fit_reducers(
    train_peptides: Sequence[Peptide], cache: EmbeddingCache, cfg: dict, dim: Optional[int] = None
) -> DualReducer:
    """Fit per-branch reducers on the pooled residues of the training peptides."""
    dual = DualReducer.from_config(cfg, dim=dim)
    p_res, e_res = _stack_residues(train_peptides, cache)
    dual.prott5.fit(p_res)
    dual.esm2.fit(e_res)
    return dual


def build_tensors(
    peptides: Sequence[Peptide],
    cache: EmbeddingCache,
    dual: DualReducer,
    window_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return padded (X_prott5, X_esm2, Y, mask) for a list of peptides."""
    n = len(peptides)
    d_p = dual.prott5.out_dim
    d_e = dual.esm2.out_dim
    X_p = np.zeros((n, window_len, d_p), dtype=np.float32)
    X_e = np.zeros((n, window_len, d_e), dtype=np.float32)
    Y = np.zeros((n, window_len), dtype=np.float32)
    mask = np.zeros((n, window_len), dtype=np.float32)

    for i, pep in enumerate(peptides):
        emb = cache.get(pep.sequence)
        rp = dual.prott5.transform(emb["prott5"])          # (Lp, d_p)
        re = dual.esm2.transform(emb["esm2"])              # (Le, d_e)
        L = min(len(pep.sequence), window_len, rp.shape[0], re.shape[0])
        X_p[i, :L] = rp[:L]
        X_e[i, :L] = re[:L]
        Y[i, :L] = np.asarray(pep.residue_labels[:L], dtype=np.float32)
        mask[i, :L] = 1.0
    return X_p, X_e, Y, mask


def residue_class_weights(Y: np.ndarray, mask: np.ndarray) -> dict:
    """Balanced per-class weights computed over *real* (unmasked) residues."""
    real = mask.astype(bool)
    labels = Y[real]
    n_pos = float(labels.sum())
    n_neg = float(labels.size - n_pos)
    total = n_pos + n_neg
    if n_pos == 0 or n_neg == 0:
        return {0: 1.0, 1: 1.0}
    return {0: total / (2 * n_neg), 1: total / (2 * n_pos)}


def sample_weights_from_mask(Y: np.ndarray, mask: np.ndarray, class_weight: dict) -> np.ndarray:
    """Combine padding mask and class weights into a per-residue sample-weight map."""
    w = np.where(Y > 0.5, class_weight[1], class_weight[0]).astype(np.float32)
    return w * mask
