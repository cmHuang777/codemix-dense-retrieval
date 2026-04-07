# Simple mMARCO Mix Router

This is a minimal, standalone router for mMARCO mix-ratio guidance. It predicts the optimal language mix ratio (alpha) for multilingual dense retrieval given a language pair and document regime.

## Quick Start

### Use the Pre-trained Model

```bash
# Predict optimal alpha for a language pair and document regime
python predict_simple_router.py --pair AR-HI --doc-mix "AR + HI docs"
# Output: 50
```

That's it! The pre-trained model is included and requires no additional dependencies.

## Model Overview

**Input features:**
- `pair`: Language pair key (e.g., "AR-HI")
- `doc_mix`: Document regime (e.g., "AR + HI docs")

**Output:**
- One alpha value from `{0, 10, 30, 50, 70, 90, 100}` representing the mix ratio

**Architecture:**
- A lightweight lookup router based on language pair and document regime
- Uses a validation-tuned smoothing with pair-level priors
- Falls back to globally optimal alpha if context is unseen

## Artifacts

The `artifacts/` directory contains the model and evaluation resources:

- `final_router/router_model.json` — The trained model (production)
- `final_router/router_table.csv` — Human-readable setting-to-alpha lookup table
- `final_router/training_summary.json` — Training, validation, and test metrics
- `final_router/eval_qids_common/` — Evaluation results on held-out test queries

## Files

- `simple_router.py` — Shared router utilities and model I/O
- `predict_simple_router.py` — Query the model for a single (pair, doc_mix)
- `train_simple_router.py` — Fit the lookup table from training data
- `train_smoothed_router.py` — Tune a regularized router on validation set
- `evaluate_simple_router.py` — Evaluate the saved router on a dataset split

## Training & Evaluation (Optional)

If you want to retrain the model or evaluate it, you need the dataset and test split definition:

```bash
# Download or prepare:
# - dataset.joblib (the training/eval dataset)
# - qids-common.tsv (the test query ID split, one qid per line)

# Then train:
python train_simple_router.py \
  --dataset /path/to/dataset.joblib \
  --qids-common /path/to/qids-common.tsv \
  --output-dir artifacts/final_router

# Optionally tune smoothing hyperparameters:
python train_smoothed_router.py \
  --dataset /path/to/dataset.joblib \
  --qids-common /path/to/qids-common.tsv \
  --output-dir artifacts/smoothed_router

# Evaluate on a split:
python evaluate_simple_router.py \
  --dataset /path/to/dataset.joblib \
  --model artifacts/final_router/router_model.json \
  --qids-common /path/to/qids-common.tsv \
  --split test \
  --output-dir artifacts/eval_qids_common
```

### Getting the Dataset

The `dataset.joblib` file is not included in this repository due to size constraints (100+ MB). To obtain it, contact the authors of the repo.

## Requirements

```
pandas
numpy
joblib
```

Install them:
```bash
pip install pandas numpy joblib
```

## Usage Examples

**Prediction only (no setup needed):**
```bash
python predict_simple_router.py --pair EN-FR --doc-mix "EN + FR docs"
python predict_simple_router.py --pair ZH-JA --doc-mix "ZH docs only"
```

**View the lookup table:**
```bash
cat artifacts/final_router/router_table.csv
```

**Training metrics:**
```bash
python -c "import json; print(json.dumps(json.load(open('artifacts/final_router/training_summary.json')), indent=2))"
```

