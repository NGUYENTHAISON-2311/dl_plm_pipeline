"""Stage 1: generate & cache per-residue ProtT5 + ESM2 embeddings.

Embeds every training peptide and every benchmark / positive-set sequence so that
later stages (training, sliding-window inference) only read from the HDF5 cache.

Usage:
    python scripts/01_generate_embeddings.py [--config config/default.yaml] [--limit N]
"""
import argparse

import _bootstrap  # noqa: F401

from src.data.dataset import load_benchmark, load_training_peptides
from src.data.embeddings import generate_embeddings
from src.utils.config import load_config, resolve_path
from src.utils.logging import get_logger
from src.utils.windows import required_inference_keys

log = get_logger("stage1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--limit", type=int, default=None, help="cap #peptides (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    peptides = load_training_peptides(
        resolve_path(cfg, "train_pos"), resolve_path(cfg, "train_neg")
    )
    if args.limit:
        peptides = peptides[: args.limit]

    # Training peptides are always embedded standalone (each peptide is its own input).
    sequences = [p.sequence for p in peptides]
    # Benchmark sequences: in full mode embed the whole sequence; in standalone mode
    # embed every sliding-window subsequence (see embeddings.embed_windows_standalone).
    for key in ("classification_benchmark", "position_benchmark"):
        try:
            bseqs = [b.sequence for b in load_benchmark(resolve_path(cfg, key))]
        except KeyError:
            continue
        sequences += required_inference_keys(bseqs, cfg)

    generate_embeddings(sequences, cfg)
    log.info("Embeddings ready in %s", cfg["embeddings"]["cache"])


if __name__ == "__main__":
    main()
