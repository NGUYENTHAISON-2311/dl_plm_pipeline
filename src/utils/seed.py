"""Global determinism helpers."""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow RNGs (TF/numpy imported lazily)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
