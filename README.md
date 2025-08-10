# NFL Game Outcome Prediction

Predicts NFL game winners using Random Forest, SVM, MLP, and XGBoost models with MLflow tracking. Run seasons and models from the command line and compare results across experiments.

## What's New vs main
- CLI args for year selection: `--train-years`, `--eval-years`, `--test-years`
- Model registry (`src/models/registry.py`) and `--models` to choose which to run
- Standardized MLflow tags: `run_type`, `split_id`, `feature_hash`, `years_*`, `models`
- Datasets saved per run under `data/processed/{run_id}/` with labels + metadata
- Comparison utility: `src/evaluation/compare_runs.py`
- Expanded metrics: ROC AUC, Log Loss, Brier Score, Calibration MAE, ECE@10/@15, MCE@10
- Other CLI args: `--experiment-name`, `--run-type`, `--split-id`, `--ece-bins`

## Repository Structure
```
src/                      # Training, evaluation, and utilities
  main_mlflow.py          # CLI entry point with MLflow integration
  models/registry.py      # Model factory
  evaluation/compare_runs.py
  ...
data/                     # Processed data and split metadata
reports/                  # Saved comparison tables
notebooks/                # Project and analysis notebooks
```

## Setup
- Python 3.10+
- `pip install -r requirements.txt`

## Data Expectations
- Place `final-<year>.csv` under `data/processed/<year>/final-<year>.csv`
- One row per matchup with `team1_*`, `team2_*`, `team1_win`

## How to Run
Baseline:
```bash
python src/main_mlflow.py \
  --experiment-name temp_exp \
  --train-years 2018,2019 \
  --eval-years 2020 \
  --test-years 2021 \
  --models rf,svm \
  --run-type baseline
```
Single-model:
```bash
python src/main_mlflow.py \
  --experiment-name temp_exp \
  --train-years 2018,2019 \
  --eval-years 2020 \
  --test-years 2021 \
  --models rf \
  --run-type kats_pack1
```
Start MLflow UI:
```bash
mlflow ui
```
Open <http://localhost:5000>

Compare recent runs:
```bash
python src/evaluation/compare_runs.py \
  --experiment-name temp_exp \
  --split-id <your_split_id> \
  --feature-hash <your_feature_hash> \
  -n 10
```
Outputs a table and saves CSV to `reports/run_comparison_YYYYMMDD_HHMMSS.csv`.

## Metrics
- Accuracy, Precision, Recall, F1
- ROC AUC (ranking quality; 0.5 random, 1.0 perfect)
- Log Loss (confident wrong predictions hurt; lower better)
- Brier Score (mean squared error of probabilities; lower better)
- Calibration MAE (global mean |p−y|)
- ECE@K (Expected Calibration Error, average bin miscalibration)
- MCE@K (Maximum Calibration Error, worst bin deviation)

Note: Platt/Isotonic calibration can improve calibration without changing ROC AUC.

## Roadmap
- Leak-free pregame features (rolling stats, Kats change-points/anomalies)
- Optional odds track
- Team ELO → Player ELO
- PBP subset (early-down success, explosive plays, pressure rate)

## Legacy README
See [README_legacy.md](README_legacy.md) for the original documentation.
