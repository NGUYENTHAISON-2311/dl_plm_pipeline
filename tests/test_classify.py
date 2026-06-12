"""The >0.5 / run>10 sequence rule."""
import numpy as np

from src.evaluation.classify import classify_profile, longest_positive_run


def test_longest_run():
    assert longest_positive_run(np.array([0, 1, 1, 0, 1])) == 2
    assert longest_positive_run(np.array([0, 0, 0])) == 0
    assert longest_positive_run(np.array([1, 1, 1])) == 3


def test_run_of_11_is_positive():
    profile = np.array([0.9] * 11 + [0.1] * 5)
    label, run, _ = classify_profile(profile, residue_threshold=0.5, min_consecutive=10)
    assert run == 11 and label == 1


def test_run_of_10_is_negative():
    # Exactly 10 consecutive is NOT > 10 -> negative.
    profile = np.array([0.9] * 10 + [0.1] * 5)
    label, run, _ = classify_profile(profile, residue_threshold=0.5, min_consecutive=10)
    assert run == 10 and label == 0


def test_threshold_applies():
    profile = np.array([0.5] * 12)  # exactly 0.5 is not > 0.5
    label, run, mask = classify_profile(profile, residue_threshold=0.5, min_consecutive=10)
    assert run == 0 and label == 0 and not mask.any()


def test_non_consecutive_does_not_count():
    profile = np.array(([0.9] * 6 + [0.1] + [0.9] * 6))  # two runs of 6, gap in middle
    label, run, _ = classify_profile(profile, residue_threshold=0.5, min_consecutive=10)
    assert run == 6 and label == 0
