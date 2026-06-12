"""Per-residue labelling, de-dup and leakage guard."""
from src.data.dataset import BenchmarkSeq, Peptide, _dedupe, assert_no_leakage


def test_residue_labels_match_sequence():
    p = Peptide(id="c1", sequence="ACDEF", label=1, group="c1")
    assert p.residue_labels == [1, 1, 1, 1, 1]
    n = Peptide(id="d1", sequence="GHIK", label=0, group="d1")
    assert n.residue_labels == [0, 0, 0, 0]


def test_dedupe_prefers_positive():
    peps = [
        Peptide(id="a", sequence="ACDE", label=0, group="a"),
        Peptide(id="b", sequence="ACDE", label=1, group="b"),
    ]
    out = _dedupe(peps)
    assert len(out) == 1 and out[0].label == 1


def test_leakage_drops_substring_peptides():
    peps = [
        Peptide(id="in", sequence="CDEF", label=1, group="in"),     # substring of bench
        Peptide(id="out", sequence="WWWW", label=0, group="out"),   # not present
    ]
    bench = [BenchmarkSeq(id="s", sequence="ABCDEFGH", label=1)]
    kept = assert_no_leakage(peps, bench, drop=True)
    ids = {p.id for p in kept}
    assert ids == {"out"}
