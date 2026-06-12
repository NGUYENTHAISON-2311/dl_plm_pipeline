# Amyloid Residue-Scoring & Classification Pipeline

Ensemble deep-learning pipeline that scores **each residue** of a protein for amyloid
propensity using frozen **ProtT5 + ESM2** embeddings (two-branch late fusion), then
derives a **sequence-level** AMYLOID / NONAMYLOID call from the residue scores.

## Tasks

1. **Residue scoring** — every residue gets a score in `[0, 1]`. Training labels are
   per-residue: amyloid core peptides → all `1`, disordered regions → all `0`.
2. **Sequence classification** — a sequence is **AMYLOID iff it contains a run of
   `>10` consecutive residues scoring `>0.5`**. This is a deterministic rule
   (`src/evaluation/classify.py`) on top of task 1, with both thresholds in config.

## Pipeline

| Stage | Script | What it does |
|------|--------|--------------|
| 1 | `scripts/01_generate_embeddings.py` | ProtT5 + ESM2 per-residue embeddings → HDF5 cache |
| 2 | `scripts/02_run_gridsearch.py` | 5-fold CV grid search per architecture (+ reducer dim) |
| 3 | `scripts/03_train_ensemble.py` | Train ensemble = best arch × 5 folds |
| 4 | `scripts/04_evaluate_benchmark.py` | Sliding-window inference + rule + metrics |

## Method summary

- **Embeddings** (`src/data/embeddings.py`): per-residue ProtT5 (1024-d) and ESM2
  (1280-d), cached once per unique sequence under `sha1(sequence)`.
- **Dimensionality reduction** (`src/data/reduce.py`): the small dataset makes 2304-d
  too wide, so a per-branch reducer (PCA by default, autoencoder optional) is **fit on
  train-fold residues only** and reused for val/benchmark to avoid leakage. Target dim
  is grid-searched.
- **Models** (`src/models/`): two-branch late fusion. Each branch (ProtT5, ESM2) is
  encoded by one of `fnn / mlp / cnn / rnn / cnn_rnn / attention`, the per-residue
  latents are concatenated, and a `TimeDistributed` sigmoid head emits one score per
  residue. Padding is ignored via masked BCE + sample weights.
- **Imbalance** (~1:9): balanced class weights by default; focal-loss / undersample
  switchable in config.
- **Ensemble** (`src/training/ensemble.py`): averages per-residue sigmoids over all
  `architecture × fold` members.
- **Inference** (`src/evaluation/sliding.py`): slide window `window_len` (stride 1) over
  each full benchmark sequence, average overlapping window scores per residue.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/01_generate_embeddings.py
python scripts/02_run_gridsearch.py
python scripts/03_train_ensemble.py
python scripts/04_evaluate_benchmark.py
# results/benchmark_report.json  +  results/profiles.json
```

### Smoke test (no GPU, tiny)

```bash
python scripts/01_generate_embeddings.py --limit 40
python scripts/02_run_gridsearch.py --models fnn --limit 40
python scripts/03_train_ensemble.py --limit 40
python scripts/04_evaluate_benchmark.py
```

### Unit tests

```bash
pip install pytest && pytest tests/
```

## Configuration

`config/default.yaml` holds paths, embedding models, reducer, CV, imbalance, ensemble
and inference settings. Each `config/models/<name>.yaml` declares an architecture's
encoder type and its hyperparameter grid. **Adding a model** = add one YAML grid + (if
a new encoder) one branch in `src/models/branches.py`.

Key inference knobs (`config/default.yaml` → `inference`):

```yaml
inference:
  window_stride: 1
  residue_threshold: 0.5     # residue positive if score > this
  min_consecutive: 10        # sequence AMYLOID if positive run length > this
  seq_overlap_agg: mean      # combine overlapping window scores: mean | max
```

## Layout

```
config/            default.yaml + models/*.yaml
src/data/          dataset, embeddings, reduce, features
src/models/        branches, heads, builders, registry
src/training/      cv, train, grid_search, ensemble
src/evaluation/    metrics, sliding, classify
scripts/           01..04 entry points
tests/             unit tests
```
