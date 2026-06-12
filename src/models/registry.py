"""Map model config names to encoder types.

Adding a new architecture = add a config/models/<name>.yaml with an ``encoder`` key
and (if it is a brand-new encoder) a branch in src/models/branches.py.
"""
from __future__ import annotations

from ..utils.config import load_model_config

# Architectures shipped with the pipeline.
AVAILABLE = ["fnn", "mlp", "cnn", "rnn", "cnn_rnn", "attention"]


def encoder_for(model_name: str) -> str:
    """Return the encoder type declared in the model's YAML config."""
    return load_model_config(model_name)["encoder"]


def grid_for(model_name: str) -> dict:
    """Return the hyperparameter grid declared in the model's YAML config."""
    return load_model_config(model_name)["grid"]
