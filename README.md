# NFL Game Outcome Prediction

Predicts NFL game winners using Random Forest, SVM, MLP, and XGBoost models with MLflow tracking. Run seasons and models from the command line and compare results across experiments.


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

## Resources to get started 

**Data Acquisition and Preparation:**
  - **Data Context/Inspiration:** This project draws inspiration and context from work in predicting NFL games:
    - [Predicting the Results of NFL Games](https://cs229.stanford.edu/proj2006/BabakHamadani-PredictingNFLGames.pdf)
    - [Predicting the Results of NFL Games Using Machine Learning (Aalto University Thesis)](https://aaltodoc.aalto.fi/server/api/core/bitstreams/80b6e0d0-f5d1-4c19-abd3-667ee40d9c93/content) 
        - The second link (Aalto University Thesis) provides context on data sources (Pro Football Focus for team statistics and scores).

- **Generating Input Data:**
    - The primary input data for the models are `final-<year>.csv` files.
    - These `final-<year>.csv` files are intended to be generated from -nitial data sources like `stats.csv` and `season.csv`.
    - Please refer to the `data/query.sql` file within this project for the SQL logic used to process and combine `stats.csv` and `season.csv` into the required `final-<year>.csv` format. (Note: The specifics of obtaining the initial `stats.csv` and `season.csv` should be based on the sources mentioned in the Aalto University Thesis).
- **Placing Data for the Project:**
    - Once you have generated the `final-<year>.csv` files, place them into appropriate subdirectories within the `data/processed/` directory. The `src/preprocessing/load_data.py` script expects this structure.
    
    - Example structure:
        ```
        data/processed/2022/final-2022.csv
        data/processed/2023/final-2023.csv
        ```

## Setup
- Python 3.10+
- Python venv
- `pip install -r requirements.txt`

## Usage
- CLI args for year selection: `--train-years`, `--eval-years`, `--test-years`
- Model registry (`src/models/registry.py`) and `--models` to choose which to run
- Standardized MLflow tags: `run_type`, `split_id`, `feature_hash`, `years_*`, `models`
- Datasets saved per run under `data/processed/{run_id}/` with labels + metadata
- Comparison utility: `src/evaluation/compare_runs.py`
- Expanded metrics: ROC AUC, Log Loss, Brier Score, Calibration MAE, ECE@10/@15, MCE@10
- Other CLI args: `--experiment-name`, `--run-type`, `--split-id`, `--ece-bins`

## Data Expectations
- Place `final-<year>.csv` under `data/processed/<year>/final-<year>.csv`
- One row per matchup with `team1_*`, `team2_*`, `team1_win`

## Models Implemented
The project trains and evaluates the following models:
* Random Forest
* Support Vector Machine (SVM)
* Multi-layer Perceptron (MLP)
* XGBoost

Configuration for these models can be found in `src/config/model_config.py`.

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
# Open <http://localhost:5000>
```

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
