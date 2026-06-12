"""Sliding-window enumeration helpers (no heavy deps).

Shared by embedding generation (scripts/01) and inference (evaluation/sliding.py) so
both agree on exactly which windows / subsequences exist for a sequence.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


def window_starts(length: int, win: int, stride: int) -> List[int]:
    """Start indices of windows covering a sequence of ``length`` residues.

    Always includes a final window flush with the end so the C-terminus is covered.
    """
    if length <= win:
        return [0]
    starts = list(range(0, length - win + 1, stride))
    if starts[-1] != length - win:
        starts.append(length - win)
    return starts


def window_subsequences(seq: str, win: int, stride: int) -> List[Tuple[int, int, str]]:
    """Return [(start, length, subsequence), ...] for every window of ``seq``."""
    L = len(seq)
    out = []
    for s in window_starts(L, win, stride):
        length = min(win, L - s)
        out.append((s, length, seq[s : s + length]))
    return out


def required_inference_keys(sequences: Sequence[str], cfg: dict) -> List[str]:
    """Sequences whose embeddings the cache must hold for sliding-window inference.

    - full mode (default): the full sequences themselves (windows slice these).
    - standalone mode (``embeddings.embed_windows_standalone: true``): every window
      subsequence, each embedded by the PLM as its own input.
    """
    if not cfg["embeddings"].get("embed_windows_standalone", False):
        return list(sequences)
    win = cfg["window_len"]
    stride = cfg["inference"]["window_stride"]
    keys: List[str] = []
    for s in sequences:
        keys.extend(sub for _, _, sub in window_subsequences(s, win, stride))
    return keys
