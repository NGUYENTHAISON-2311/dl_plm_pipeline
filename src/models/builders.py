"""Assemble two-branch late-fusion Keras models for per-residue scoring."""
from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras import Input, Model, optimizers

from .branches import build_branch_encoder
from .heads import fuse_and_classify


def masked_bce(y_true, y_pred):
    """Binary cross-entropy that ignores padded residues via sample weights.

    Padding is handled through ``sample_weight`` passed to ``fit``; this loss simply
    clips probabilities for numerical stability and averages per residue.
    """
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return bce


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1.0 - y_pred)
        a = tf.where(tf.equal(y_true, 1), alpha, 1.0 - alpha)
        return -a * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)

    return loss


def build_model(
    encoder: str,
    hp: Dict[str, Any],
    window_len: int,
    dim_prott5: int,
    dim_esm2: int,
    learning_rate: float = 1e-3,
    imbalance: Dict[str, Any] | None = None,
) -> Model:
    """Build, compile and return a two-branch per-residue scoring model."""
    in_p = Input(shape=(window_len, dim_prott5), name="prott5_in")
    in_e = Input(shape=(window_len, dim_esm2), name="esm2_in")

    enc_p = build_branch_encoder(in_p, encoder, hp, name="p")
    enc_e = build_branch_encoder(in_e, encoder, hp, name="e")

    out = fuse_and_classify(enc_p, enc_e, hp)
    model = Model([in_p, in_e], out, name=f"{encoder}_dualbranch")

    imbalance = imbalance or {}
    if imbalance.get("strategy") == "focal":
        loss = focal_loss(imbalance.get("focal_gamma", 2.0), imbalance.get("focal_alpha", 0.25))
    else:
        loss = masked_bce

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        weighted_metrics=[
            tf.keras.metrics.AUC(name="auroc", curve="ROC"),
            tf.keras.metrics.AUC(name="auprc", curve="PR"),
        ],
    )
    return model
