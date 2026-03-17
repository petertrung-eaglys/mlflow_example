# MLflow Basic — Fraud Detection on YelpChi

MLflow-tracked ML experimentation for fraud detection on the [YelpChi](https://github.com/YingtongDou/CARE-GNN) dataset. Compares classical and deep learning models on a graph-structured dataset (~45k nodes, 32 features, binary fraud labels). This example was taken from the book [Graph Neural Network in Action](https://www.manning.com/books/graph-neural-networks-in-action) and modified.

## Setup

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

**Setting up MLflow for tracking**

There are three different options to run MLflow for tracking, as shown on the figure below

![Tracking options for MLflow](doc/tracking-setup-overview.png)

**Configure MLflow environment variables:**

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

# GCN (PyTorch Geometric)
uv run python src/gcn.py
uv run python src/gcn.py --epochs 1000 --patience 50 --hidden-sizes 64 32

# GAT (PyTorch Geometric)
uv run python src/gat.py
uv run python src/gat.py --epochs 1000 --patience 50 --hidden-sizes 64 32 --heads 4
```

All scripts accept `--matrix` and `--adjlist` to override default data paths.

### Logistic Regression options

| Flag | Default | Description |
|---|---|---|
| `--thres` | 0.5 | Classification threshold |
| `--random-state` | 0 | Random seed |

### XGBoost options

| Flag | Default | Description |
|---|---|---|
| `--n-estimators` | 100 | Number of boosting rounds |
| `--max-depth` | 6 | Maximum tree depth |
| `--lr` | 0.3 | Boosting learning rate (eta) |
| `--subsample` | 1.0 | Subsample ratio of training instances |
| `--colsample-bytree` | 1.0 | Subsample ratio of columns per tree |
| `--thres` | 0.5 | Classification threshold |

### MLP options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 10000 | Maximum training epochs |
| `--patience` | 100 | Early stopping patience |
| `--hidden-sizes` | 64 | Hidden layer sizes, e.g. `--hidden-sizes 64 32` |
| `--lr` | 0.001 | Learning rate |
| `--thres` | 0.5 | Classification threshold |

### GCN options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 10000 | Maximum training epochs |
| `--patience` | 100 | Early stopping patience |
| `--hidden-sizes` | 64 | Hidden layer sizes, e.g. `--hidden-sizes 64 32` |
| `--lr` | 0.001 | Learning rate |
| `--thres` | 0.5 | Classification threshold |

### GAT options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 10000 | Maximum training epochs |
| `--patience` | 100 | Early stopping patience |
| `--batch-size` | 256 | Batch size for neighbour sampling |
| `--hidden-sizes` | 32 | Hidden layer sizes, e.g. `--hidden-sizes 64 32` |
| `--heads` | 1 | Number of attention heads per GAT layer |
| `--lr` | 0.001 | Learning rate |
| `--thres` | 0.5 | Classification threshold |

## Models

| Script | Model | Framework |
|---|---|---|
| `src/logres.py` | Logistic Regression | scikit-learn |
| `src/xgb.py` | XGBoost | xgboost |
| `src/mlp.py` | MLP with configurable hidden layers and early stopping | PyTorch |
| `src/gcn.py` | Graph Convolutional Network with configurable hidden layers | PyTorch Geometric |
| `src/gat.py` | Graph Attention Network with configurable hidden layers and attention heads | PyTorch Geometric |

## Metrics

All models report: **ROC AUC**, **F1-score**, **Precision**, **Recall** (threshold 0.5 for binary predictions). Metrics are logged to MLflow for each run.

## Development

```bash
# Lint and format
uv run ruff check src/
uv run ruff format src/
```
