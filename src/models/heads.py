"""Late-fusion + per-residue classifier head."""
from __future__ import annotations

from typing import Any, Dict

from tensorflow.keras import layers


def fuse_and_classify(branch_p, branch_e, hp: Dict[str, Any]):
    """Concatenate the two per-residue branch latents and emit one score per residue.

    Inputs are (L, h_p) and (L, h_e); output is (L, 1) sigmoid scores.
    """
    dropout = hp.get("head_dropout", hp.get("dropout", 0.3))
    head_units = hp.get("head_units", 64)
    x = layers.Concatenate(name="fusion")([branch_p, branch_e])
    x = layers.TimeDistributed(layers.Dense(head_units, activation="relu"), name="head_dense")(x)
    x = layers.Dropout(dropout, name="head_drop")(x)
    out = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"), name="residue_score")(x)
    return layers.Reshape((-1,), name="squeeze")(out)  # (L,) per-residue probabilities
