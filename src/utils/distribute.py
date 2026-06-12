"""Optional multi-GPU training via tf.distribute.MirroredStrategy.

Enabled with ``train.distribute: mirrored`` in the config (default ``none``). On a
single-GPU or CPU host this is a no-op. Note: the models here are small and the dataset
tiny, so multi-GPU rarely speeds training up — the bigger 2x-GPU win is parallel
embedding generation (see embeddings.generate_embeddings_dual_gpu).
"""
from __future__ import annotations

import contextlib

from .logging import get_logger

log = get_logger(__name__)

_STRATEGY = None
_RESOLVED = False


def get_strategy(cfg: dict):
    """Return a cached MirroredStrategy if requested and >1 GPU is visible, else None."""
    global _STRATEGY, _RESOLVED
    if cfg.get("train", {}).get("distribute", "none") != "mirrored":
        return None
    if not _RESOLVED:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 1:
            _STRATEGY = tf.distribute.MirroredStrategy()
            log.info("MirroredStrategy enabled across %d GPUs", _STRATEGY.num_replicas_in_sync)
        else:
            log.info("train.distribute=mirrored but only %d GPU visible; running single-device", len(gpus))
            _STRATEGY = None
        _RESOLVED = True
    return _STRATEGY


def build_scope(cfg: dict):
    """Context manager for model construction: strategy scope if enabled, else a no-op."""
    strat = get_strategy(cfg)
    return strat.scope() if strat is not None else contextlib.nullcontext()
