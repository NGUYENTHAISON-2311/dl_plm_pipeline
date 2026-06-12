"""Per-branch residue encoders.

Each encoder maps a per-residue feature sequence ``(L, d)`` to a per-residue latent
sequence ``(L, h)`` (i.e. it preserves the residue axis so the final head can emit one
score per residue). The same encoder factory is applied independently to the ProtT5
and ESM2 branches before late fusion.
"""
from __future__ import annotations

from typing import Any, Dict

from tensorflow.keras import layers


def build_branch_encoder(x, encoder: str, hp: Dict[str, Any], name: str):
    """Apply an encoder of the given type to tensor ``x`` of shape (L, d)."""
    if encoder == "dense":
        return _dense(x, hp, name)
    if encoder == "cnn":
        return _cnn(x, hp, name)
    if encoder == "rnn":
        return _rnn(x, hp, name)
    if encoder == "cnn_rnn":
        return _cnn_rnn(x, hp, name)
    if encoder == "attention":
        return _attention(x, hp, name)
    raise ValueError(f"Unknown encoder: {encoder}")


def _dense(x, hp, name):
    """TimeDistributed dense stack — scores each residue independently (FNN/MLP)."""
    units = hp.get("units", [64])
    dropout = hp.get("dropout", 0.3)
    for i, u in enumerate(units):
        x = layers.TimeDistributed(layers.Dense(u, activation="relu"), name=f"{name}_dense{i}")(x)
        x = layers.TimeDistributed(layers.BatchNormalization(), name=f"{name}_bn{i}")(x)
        x = layers.Dropout(dropout, name=f"{name}_drop{i}")(x)
    return x


def _cnn(x, hp, name):
    """Stacked Conv1D (same padding) — local motif detector, residue axis preserved."""
    filters = hp.get("filters", [64, 64])
    k = hp.get("kernel_size", 3)
    dropout = hp.get("dropout", 0.3)
    for i, f in enumerate(filters):
        x = layers.Conv1D(f, k, padding="same", activation="relu", name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Dropout(dropout, name=f"{name}_drop{i}")(x)
    return x


def _rnn(x, hp, name):
    """Bidirectional RNN with return_sequences -> per-residue context."""
    rnn_type = hp.get("rnn_type", "lstm")
    units = hp.get("units", 64)
    dropout = hp.get("dropout", 0.3)
    cell = layers.LSTM if rnn_type == "lstm" else layers.GRU
    x = layers.Bidirectional(cell(units, return_sequences=True), name=f"{name}_bi{rnn_type}")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def _cnn_rnn(x, hp, name):
    """Conv1D (local) -> BiLSTM (global), residue axis preserved throughout."""
    x = _cnn(x, {**hp, "filters": hp.get("filters", [64])}, f"{name}_c")
    x = _rnn(x, hp, f"{name}_r")
    return x


def _attention(x, hp, name):
    """Conv1D -> MultiHeadSelfAttention (transformer-lite), residue axis preserved."""
    x = _cnn(x, {**hp, "filters": hp.get("filters", [64])}, f"{name}_c")
    num_heads = hp.get("num_heads", 2)
    key_dim = hp.get("key_dim", 32)
    dropout = hp.get("dropout", 0.3)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f"{name}_mha")(x, x)
    x = layers.Add(name=f"{name}_res")([x, attn])
    x = layers.LayerNormalization(name=f"{name}_ln")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x
