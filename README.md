# MLflow Basic — Fraud Detection on YelpChi

MLflow-tracked ML experimentation for fraud detection on the [YelpChi](https://github.com/YingtongDou/CARE-GNN) dataset. Compares classical and deep learning models on a graph-structured dataset (~45k nodes, 32 features, binary fraud labels).

## Setup

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

**Configure MLflow:**

```bash
cp .env.example .env
# Edit .env with your MLflow tracking URI and experiment name
```

`.env` variables:

| Variable | Description | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000/` |
| `EXPERIMENT_NAME` | MLflow experiment name | `yelp_review` |

**Dataset:** Unzip `YelpChi.zip` in `data/`. The adjacency list pickle is generated automatically on first run.

## Running Models

All scripts must be run from the repo root so `data/` paths resolve correctly.

```bash
# Logistic Regression
uv run python src/logres.py

# XGBoost
uv run python src/xgb.py

# MLP (PyTorch)
uv run python src/mlp.py
uv run python src/mlp.py --epochs 1000 --patience 50 --hidden-sizes 64 32 --lr 0.001
```

All scripts accept `--matrix` and `--adjlist` to override default data paths.

### MLP options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 10000 | Maximum training epochs |
| `--patience` | 100 | Early stopping patience |
| `--hidden-sizes` | `64` | Hidden layer sizes, e.g. `--hidden-sizes 64 32` |
| `--lr` | 0.001 | Learning rate |

### Logistic Regression options

| Flag | Default | Description |
|---|---|---|
| `--thres` | 0.5 | Classification threshold |
| `--random-state` | 0 | Random seed |

## Models

| Script | Model | Framework |
|---|---|---|
| `src/logres.py` | Logistic Regression | scikit-learn |
| `src/xgb.py` | XGBoost | xgboost |
| `src/mlp.py` | MLP (32→64→1) with early stopping | PyTorch |
| `src/gcn.py` | Graph Convolutional Network *(in progress)* | PyTorch Geometric |

## Metrics

All models report: **ROC AUC**, **F1-score**, **Precision**, **Recall** (threshold 0.5 for binary predictions). Metrics are logged to MLflow for each run.

## Development

```bash
# Lint and format
uv run ruff check src/
uv run ruff format src/
```
