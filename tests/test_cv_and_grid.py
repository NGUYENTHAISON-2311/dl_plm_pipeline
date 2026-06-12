"""Grid expansion and CV fold disjointness/stratification."""
import numpy as np

from src.data.dataset import Peptide
from src.training.cv import make_folds
from src.training.grid_search import expand_grid


def test_expand_grid_cartesian():
    grid = {"a": [1, 2], "b": ["x", "y", "z"]}
    combos = expand_grid(grid)
    assert len(combos) == 6
    assert {"a": 1, "b": "x"} in combos and {"a": 2, "b": "z"} in combos


def _toy_peptides(n_pos=20, n_neg=40):
    peps = [Peptide(id=f"p{i}", sequence="ACDE", label=1, group=f"p{i}") for i in range(n_pos)]
    peps += [Peptide(id=f"n{i}", sequence="GHIK", label=0, group=f"n{i}") for i in range(n_neg)]
    return peps


def test_folds_disjoint_and_cover():
    peps = _toy_peptides()
    cfg = {"seed": 0, "cv": {"scheme": "stratified", "n_splits": 5, "shuffle": True}}
    folds = make_folds(peps, cfg)
    assert len(folds) == 5
    all_val = []
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va))            # disjoint within a fold
        all_val.extend(va.tolist())
    assert sorted(all_val) == list(range(len(peps)))  # val folds partition the set


def test_folds_are_stratified():
    peps = _toy_peptides()
    labels = np.array([p.label for p in peps])
    cfg = {"seed": 0, "cv": {"scheme": "stratified", "n_splits": 5, "shuffle": True}}
    overall = labels.mean()
    for _, va in make_folds(peps, cfg):
        # each validation fold roughly preserves the positive ratio
        assert abs(labels[va].mean() - overall) < 0.15
