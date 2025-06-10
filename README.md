# Photo‑Tag Pipeline

A modular, reproducible photo‑tagging ML pipeline with:

* **src/** code‑only package layout
* **MLflow** experiment tracking
* **GitHub Actions** CI (lint → tests → smoke train → eval)
* Quick synthetic dataset for CI smoke tests
* Placeholder hooks for future DVC data/model versioning

## 1 — Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2 — Run locally

```bash
python scripts/run_train.py --epochs 3                # train
python src/evaluate.py                                # evaluate
python scripts/run_pipeline.py                        # train + eval
```

## 3 — Experiment tracking

Launch the MLflow UI:

```bash
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file:mlruns}
mlflow ui
```

Open <http://127.0.0.1:5000> to browse runs, metrics and artifacts.

## 4 — CI

Every push / PR to **main** triggers `.github/workflows/ci.yml`:

* **black**, **isort**, **flake8**
* **pytest** unit tests
* 1‑epoch smoke‑train + evaluation
* Uploads artifacts (`checkpoints/`, `results/`, `mlruns/`)

See the workflow for details.

---

> **Note** : Full datasets & long training jobs should run outside CI.  
> Data/model versioning via **DVC** can be added later.
