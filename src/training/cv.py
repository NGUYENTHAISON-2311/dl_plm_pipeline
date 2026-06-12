"""Cross-validation fold generation over peptides."""
from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import numpy as np

from ..data.dataset import Peptide


def make_folds(
    peptides: Sequence[Peptide], cfg: dict
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, val_idx) arrays for the configured CV scheme.

    - ``stratified`` (default): StratifiedKFold on peptide label.
    - ``group``: StratifiedGroupKFold grouping by ``core_id`` to prevent regions from
      the same cluster spanning folds.
    """
    cv = cfg["cv"]
    n = cv["n_splits"]
    seed = cfg.get("seed", 42)
    labels = np.array([p.label for p in peptides])

    if cv.get("group_by_core", False) or cv.get("scheme") == "group":
        from sklearn.model_selection import StratifiedGroupKFold

        groups = np.array([p.group for p in peptides])
        splitter = StratifiedGroupKFold(n_splits=n, shuffle=cv.get("shuffle", True), random_state=seed)
        return list(splitter.split(np.zeros(len(peptides)), labels, groups))

    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=n, shuffle=cv.get("shuffle", True), random_state=seed)
    return list(splitter.split(np.zeros(len(peptides)), labels))


def subset(peptides: Sequence[Peptide], idx: np.ndarray) -> List[Peptide]:
    return [peptides[i] for i in idx]
