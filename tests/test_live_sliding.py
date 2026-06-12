"""Live sliding-window benchmark scoring (PLM-on-each-window), using stubs."""
import numpy as np

from src.evaluation.sliding import score_sequence_live


class _StubEmbedder:
    """Returns zero embeddings sized to each window (PLM stand-in)."""

    def embed_many(self, seqs, batch_size=64):
        return [{"prott5": np.zeros((len(s), 4), "float32"),
                 "esm2": np.zeros((len(s), 5), "float32")} for s in seqs]


class _StubEnsemble:
    window_len = 18
    members = [1, 2, 3]

    def __init__(self, value=0.8):
        self.value = value

    def predict_window_batch(self, emb_list):
        return [np.full(e["prott5"].shape[0], self.value) for e in emb_list]


def _cfg(win=18, stride=1, agg="mean"):
    return {"window_len": win,
            "inference": {"window_stride": stride, "seq_overlap_agg": agg, "plm_batch_size": 64}}


def test_window_count_and_coverage():
    prof, wins = score_sequence_live("A" * 40, _StubEmbedder(), _StubEnsemble(0.8), _cfg())
    assert len(prof) == 40
    assert len(wins) == 40 - 18 + 1            # stride-1 windows
    assert wins[-1]["end"] == 40               # last window flush with the C-terminus
    assert np.allclose(prof, 0.8)              # mean of constant window scores


def test_short_sequence_single_window():
    prof, wins = score_sequence_live("A" * 10, _StubEmbedder(), _StubEnsemble(0.5), _cfg())
    assert len(wins) == 1 and wins[0]["start"] == 0 and wins[0]["end"] == 10
    assert len(prof) == 10


def test_max_aggregation_picks_highest_overlap():
    # Two windows overlap; with max-agg the profile takes the larger contributor.
    emb = _StubEmbedder()
    ens = _StubEnsemble(0.9)
    prof, _ = score_sequence_live("A" * 20, emb, ens, _cfg(agg="max"))
    assert np.allclose(prof, 0.9)
