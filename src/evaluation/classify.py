"""Sequence-level decision rule on top of a per-residue score profile.

A residue is positive if its score exceeds ``residue_threshold`` (0.5). A sequence is
AMYLOID iff the longest run of consecutive positive residues exceeds ``min_consecutive``
(default 10, i.e. a run of >10 / at least 11). Both parameters come from config so the
rule stays tunable.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def longest_positive_run(positive_mask: np.ndarray) -> int:
    """Length of the longest run of consecutive True values."""
    best = run = 0
    for v in positive_mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def classify_profile(
    profile: np.ndarray, residue_threshold: float = 0.5, min_consecutive: int = 10
) -> Tuple[int, int, np.ndarray]:
    """Apply the rule to a per-residue score profile.

    Returns (label, longest_run, positive_mask) where ``label`` is 1 (AMYLOID) iff
    ``longest_run > min_consecutive``.
    """
    positive_mask = profile > residue_threshold
    run = longest_positive_run(positive_mask)
    label = int(run > min_consecutive)
    return label, run, positive_mask


def sequence_score(profile: np.ndarray, min_consecutive: int = 10) -> float:
    """Continuous sequence score in [0,1] consistent with the >threshold/run rule.

    A sequence is positive iff some window of ``w = min_consecutive + 1`` consecutive
    residues are all above threshold. The best such window's *minimum* score is the
    largest threshold that would still call the sequence positive, so it is a natural
    continuous score: ``sequence_score(profile) > t`` exactly reproduces the rule with
    ``residue_threshold = t``. Useful for ROC/PR/AUROC at the sequence level.
    """
    w = min_consecutive + 1
    p = np.asarray(profile, dtype=np.float64)
    if p.size < w:
        return float(p.min()) if p.size else 0.0
    # max over windows of the per-window minimum score
    best = 0.0
    for s in range(p.size - w + 1):
        best = max(best, float(p[s : s + w].min()))
    return best
