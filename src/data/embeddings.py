"""Per-residue ProtT5 + ESM2 embedding generation with an HDF5 cache.

Embeddings are frozen features, so generation uses HuggingFace/PyTorch even though
the downstream models are Keras. Each unique sequence is embedded once and stored
under ``sha1(sequence)`` with datasets ``prott5`` (L,1024) and ``esm2`` (L,1280).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np

from ..utils.config import PROJECT_ROOT
from ..utils.logging import get_logger

log = get_logger(__name__)


def seq_key(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class EmbeddingCache:
    """Read/write access to the HDF5 embedding store."""

    def __init__(self, path: str | Path):
        # Resolve relative cache paths against the project root so the cache is found
        # regardless of the current working directory (e.g. when run from notebooks/).
        p = Path(path)
        self.path = p if p.is_absolute() else PROJECT_ROOT / p
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def has(self, sequence: str) -> bool:
        if not self.path.exists():
            return False
        with h5py.File(self.path, "r") as f:
            grp = f.get(seq_key(sequence))
            return grp is not None and "prott5" in grp and "esm2" in grp

    def get(self, sequence: str) -> Dict[str, np.ndarray]:
        with h5py.File(self.path, "r") as f:
            grp = f[seq_key(sequence)]
            return {"prott5": grp["prott5"][:], "esm2": grp["esm2"][:]}

    def put(self, sequence: str, prott5: np.ndarray, esm2: np.ndarray) -> None:
        with h5py.File(self.path, "a") as f:
            key = seq_key(sequence)
            if key in f:
                del f[key]
            grp = f.create_group(key)
            grp.attrs["sequence"] = sequence
            grp.create_dataset("prott5", data=prott5.astype(np.float32), compression="gzip")
            grp.create_dataset("esm2", data=esm2.astype(np.float32), compression="gzip")

    def missing(self, sequences: Iterable[str]) -> List[str]:
        uniq = list(dict.fromkeys(sequences))
        return [s for s in uniq if not self.has(s)]


# --------------------------------------------------------------------------- #
# Encoders (lazy-loaded so the rest of the pipeline imports without torch)
# --------------------------------------------------------------------------- #
class _ProtT5Encoder:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import T5EncoderModel, T5Tokenizer

        self.torch = torch
        self.device = device
        self.tok = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_name).to(device).eval()

    def embed(self, sequence: str) -> np.ndarray:
        return self.embed_batch([sequence])[0]

    def embed_batch(self, sequences: List[str]) -> List[np.ndarray]:
        """Embed several sequences in ONE padded forward pass (GPU-efficient)."""
        # ProtT5 expects space-separated residues with rare AAs mapped to X.
        spaced = [" ".join(re.sub(r"[UZOB]", "X", s)) for s in sequences]
        enc = self.tok(spaced, return_tensors="pt", padding=True, add_special_tokens=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model(**enc).last_hidden_state  # (B, Lmax, 1024)
        out = out.cpu().numpy()
        # Real residue tokens come first; the trailing </s> and padding are dropped by
        # keeping exactly len(seq) positions per row.
        return [out[i, : len(s)] for i, s in enumerate(sequences)]


class _ESM2Encoder:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def embed(self, sequence: str) -> np.ndarray:
        return self.embed_batch([sequence])[0]

    def embed_batch(self, sequences: List[str]) -> List[np.ndarray]:
        """Embed several sequences in ONE padded forward pass (GPU-efficient)."""
        enc = self.tok(list(sequences), return_tensors="pt", padding=True, add_special_tokens=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model(**enc).last_hidden_state  # (B, Lmax, 1280)
        out = out.cpu().numpy()
        # ESM2 adds <cls> at position 0 and <eos> after the sequence; residues are 1..L.
        return [out[i, 1 : len(s) + 1] for i, s in enumerate(sequences)]


def generate_embeddings(sequences: Iterable[str], cfg: dict) -> None:
    """Populate the cache for any sequence not already present."""
    cache = EmbeddingCache(cfg["embeddings"]["cache"])
    todo = cache.missing(sequences)
    if not todo:
        log.info("Embedding cache already complete (%d sequences).",
                 len(list(dict.fromkeys(sequences))))
        return

    device = _resolve_device(cfg["embeddings"].get("device", "auto"))
    log.info("Generating embeddings for %d sequences on %s", len(todo), device)

    prott5 = _ProtT5Encoder(cfg["embeddings"]["prott5"]["model"], device)
    esm2 = _ESM2Encoder(cfg["embeddings"]["esm2"]["model"], device)

    for i, seq in enumerate(todo, 1):
        p = prott5.embed(seq)
        e = esm2.embed(seq)
        if p.shape[0] != len(seq) or e.shape[0] != len(seq):
            log.warning("Length mismatch for %s: prott5=%s esm2=%s seq=%d",
                        seq[:12], p.shape, e.shape, len(seq))
        cache.put(seq, p, e)
        if i % 50 == 0 or i == len(todo):
            log.info("  embedded %d/%d", i, len(todo))


def generate_embeddings_dual_gpu(
    sequences: Iterable[str], cfg: dict,
    device_prott5: str = "cuda:0", device_esm2: str = "cuda:1",
) -> None:
    """Populate the cache using TWO GPUs in parallel (e.g. Kaggle 2x T4).

    ProtT5 runs on one GPU and ESM2 on the other, in separate threads. PyTorch releases
    the GIL during CUDA ops, so both GPUs are busy at the same time. Embeddings are held
    in memory until both models finish, then written to the HDF5 cache by the main thread
    (single writer -> no concurrent-write corruption).
    """
    import threading

    cache = EmbeddingCache(cfg["embeddings"]["cache"])
    todo = cache.missing(sequences)
    if not todo:
        log.info("Embedding cache already complete (%d sequences).",
                 len(list(dict.fromkeys(sequences))))
        return

    log.info("Dual-GPU embedding: %d sequences (ProtT5->%s, ESM2->%s)",
             len(todo), device_prott5, device_esm2)
    prott5_out: Dict[str, np.ndarray] = {}
    esm2_out: Dict[str, np.ndarray] = {}
    errors: List[Exception] = []

    def _worker(encoder_factory, store, tag):
        try:
            enc = encoder_factory()
            for i, seq in enumerate(todo, 1):
                store[seq] = enc.embed(seq)
                if i % 50 == 0 or i == len(todo):
                    log.info("  %s %d/%d", tag, i, len(todo))
        except Exception as exc:  # surface worker failure to the caller
            errors.append(exc)

    t_p = threading.Thread(target=_worker, args=(
        lambda: _ProtT5Encoder(cfg["embeddings"]["prott5"]["model"], device_prott5),
        prott5_out, "ProtT5"))
    t_e = threading.Thread(target=_worker, args=(
        lambda: _ESM2Encoder(cfg["embeddings"]["esm2"]["model"], device_esm2),
        esm2_out, "ESM2"))
    t_p.start(); t_e.start(); t_p.join(); t_e.join()
    if errors:
        raise errors[0]

    for seq in todo:
        cache.put(seq, prott5_out[seq], esm2_out[seq])
    log.info("Dual-GPU embedding complete -> %s", cache.path)


class PLMEmbedder:
    """Runs ProtT5 + ESM2 **live** on sequences/windows (no cache), GPU-batched.

    Used by the benchmark so the PLMs perform inference directly on each sliding window
    (e.g. 18-residue windows) instead of slicing precomputed full-sequence embeddings.
    ProtT5 and ESM2 may sit on different GPUs (e.g. Kaggle 2x T4) via the device args.
    """

    def __init__(self, cfg: dict, device: str = "auto",
                 device_prott5: str | None = None, device_esm2: str | None = None):
        dp = device_prott5 or _resolve_device(device)
        de = device_esm2 or _resolve_device(device)
        log.info("PLMEmbedder: ProtT5->%s, ESM2->%s", dp, de)
        self.prott5 = _ProtT5Encoder(cfg["embeddings"]["prott5"]["model"], dp)
        self.esm2 = _ESM2Encoder(cfg["embeddings"]["esm2"]["model"], de)

    def embed_many(self, sequences: List[str], batch_size: int = 64) -> List[Dict[str, np.ndarray]]:
        """Embed many (typically equal-length window) sequences in batches.

        Returns one dict per input: ``{"prott5": (L,1024), "esm2": (L,1280)}``.
        """
        seqs = list(sequences)
        out: List[Dict[str, np.ndarray]] = []
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i : i + batch_size]
            p_list = self.prott5.embed_batch(chunk)
            e_list = self.esm2.embed_batch(chunk)
            out.extend({"prott5": p, "esm2": e} for p, e in zip(p_list, e_list))
        return out
