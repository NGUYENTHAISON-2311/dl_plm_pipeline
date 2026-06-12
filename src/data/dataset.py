"""Load raw JSON datasets into per-residue labelled peptide examples.

Each training example is one peptide (a short region) carrying a per-residue label
vector: amyloid cores -> all ones, disordered regions -> all zeros. This matches the
residue-scoring task; the sequence-level call is derived later by the >0.5 / run>10
rule in ``src.evaluation.classify``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..utils.io import read_json
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Peptide:
    """A labelled training region."""

    id: str
    sequence: str
    label: int                       # peptide-level class (1 core / 0 disordered)
    group: str                       # core_id, used for group-stratified CV
    residue_labels: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.residue_labels:
            self.residue_labels = [self.label] * len(self.sequence)


@dataclass
class BenchmarkSeq:
    """A full-length benchmark sequence with optional core annotations."""

    id: str
    sequence: str
    label: int                       # 1 AMYLOID / 0 NONAMYLOID
    core_regions: List[Dict[str, Any]] = field(default_factory=list)


def load_training_peptides(pos_path: str | Path, neg_path: str | Path) -> List[Peptide]:
    """Build the per-residue labelled training set from cores (+1) and disordered (0)."""
    peptides: List[Peptide] = []

    for item in read_json(pos_path):
        seq = item["sequence"].strip().upper()
        cid = item.get("core_id", item.get("id", f"pos_{len(peptides)}"))
        peptides.append(Peptide(id=str(cid), sequence=seq, label=1, group=str(cid)))

    for item in read_json(neg_path):
        seq = item["sequence"].strip().upper()
        cid = item.get("core_id", item.get("id", f"neg_{len(peptides)}"))
        peptides.append(Peptide(id=str(cid), sequence=seq, label=0, group=str(cid)))

    peptides = _dedupe(peptides)
    n_pos = sum(p.label for p in peptides)
    log.info("Loaded %d peptides (%d positive / %d negative)",
             len(peptides), n_pos, len(peptides) - n_pos)
    return peptides


def load_benchmark(path: str | Path) -> List[BenchmarkSeq]:
    """Load the full-length classification benchmark."""
    out: List[BenchmarkSeq] = []
    for item in read_json(path):
        label = 1 if str(item.get("LABEL", "")).upper() == "AMYLOID" else 0
        out.append(
            BenchmarkSeq(
                id=item.get("ID", f"seq_{len(out)}"),
                sequence=item["Sequence"].strip().upper(),
                label=label,
                core_regions=item.get("matched_core_regions", []) or [],
            )
        )
    n_pos = sum(s.label for s in out)
    log.info("Loaded %d benchmark sequences (%d AMYLOID / %d NONAMYLOID)",
             len(out), n_pos, len(out) - n_pos)
    return out


def _dedupe(peptides: List[Peptide]) -> List[Peptide]:
    """Drop exact-duplicate sequences, keeping the first occurrence.

    Logs any sequence that appears with conflicting labels (positive wins).
    """
    seen: Dict[str, Peptide] = {}
    for p in peptides:
        prev = seen.get(p.sequence)
        if prev is None:
            seen[p.sequence] = p
        elif prev.label != p.label:
            log.warning("Conflicting labels for %s; keeping positive", p.sequence)
            if p.label == 1:
                seen[p.sequence] = p
    dropped = len(peptides) - len(seen)
    if dropped:
        log.info("De-duplicated %d exact-duplicate peptides", dropped)
    return list(seen.values())


def assert_no_leakage(
    peptides: List[Peptide], benchmark: List[BenchmarkSeq], drop: bool = True
) -> List[Peptide]:
    """Guard against train/benchmark overlap.

    A training peptide leaks if its sequence is a substring of any benchmark
    sequence. Such peptides are logged and (by default) dropped from training.
    """
    bench_seqs = [b.sequence for b in benchmark]
    kept, leaked = [], []
    for p in peptides:
        if any(p.sequence in bseq for bseq in bench_seqs):
            leaked.append(p)
        else:
            kept.append(p)
    if leaked:
        log.warning("Leakage: %d training peptides are substrings of benchmark "
                    "sequences (e.g. %s)", len(leaked), leaked[0].sequence)
    if not drop:
        return peptides
    return kept
