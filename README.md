# NFL Game Outcome Prediction Project

This project uses machine learning to predict the outcome of NFL games based on historical game statistics. It employs several models and uses MLflow for experiment tracking and reproducibility.

## Project Structure
```
NFL-ML/
├── data/
│   ├── query.sql         # SQL query to generate final-<year>.csv 
│   ├── raw/              # (Optional) For initial stats.csv, season.csv
│   └── processed/        # For final-<year>.csv files
├── notebooks/          # (Optional) Jupyter notebooks for EDA
├── src/
│   ├── config/
│   │   └── model_config.py
│   ├── evaluation/
│   │   ├── metrics_calculator.py
│   │   └── plotting.py
│   ├── mlruns/         # MLflow tracking data (auto-generated)
│   ├── models/
│   │   ├── random_forest_model.py
│   │   ├── svm_model.py
│   │   ├── mlp_model.py
│   │   └── xgboost_model.py
│   ├── preprocessing/
│   │   ├── load_data.py
│   │   └── prepare_data.py
│   └── main_mlflow.py    # Main script with MLflow integration 
│   └── main.py           # Simple starter main script without ML Flow. 
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup, Info
1.  **Install Dependencies:**
* It is highly recommended to use a Python virtual environment for this project.
* Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Acquisition and Preparation:**
    * **Data Context/Inspiration:** This project draws inspiration and context from work in predicting NFL games:
        * [Predicting the Results of NFL Games](https://cs229.stanford.edu/proj2006/BabakHamadani-PredictingNFLGames.pdf)
        * [Predicting the Results of NFL Games Using Machine Learning (Aalto University Thesis)](https://aaltodoc.aalto.fi/server/api/core/bitstreams/80b6e0d0-f5d1-4c19-abd3-667ee40d9c93/content) 
            * The second link (Aalto University Thesis) provides context on data sources (Pro Football Focus for team statistics and scores).
    * **Generating Input Data:**
        * The primary input data for the models are `final-<year>.csv` files.
        * These `final-<year>.csv` files are intended to be generated from initial data sources like `stats.csv` and `season.csv`.
        * Please refer to the `data/query.sql` file within this project for the SQL logic used to process and combine `stats.csv` and `season.csv` into the required `final-<year>.csv` format. (Note: The specifics of obtaining the initial `stats.csv` and `season.csv` should be based on the sources mentioned in the Aalto University Thesis).
    * **Placing Data for the Project:**
        * Once you have generated the `final-<year>.csv` files, place them into appropriate subdirectories within the `data/processed/` directory. The `src/preprocessing/load_data.py` script expects this structure.
        * Example structure:
            ```
            data/processed/2022/final-2022.csv
            data/processed/2023/final-2023.csv
            ```


## Running Experiments
The main script for running experiments with MLflow tracking is `src/main_mlflow.py`.
1.  **Navigate to the `src` directory and run from the project root:**
    ```bash
    cd src && python src/main_mlflow.py
    or 
    python src/main_mlflow.py
    ```
2.  **Specify an MLflow Experiment Name (Optional):**
    You can specify an experiment name. If not provided, it defaults to "NFL\_Game\_Outcome\_Prediction".
    ```bash
    python src/main_mlflow.py --experiment-name "My_NFL_Experiments_Phase_1"
    ```
3.  **Viewing Results with MLflow UI:**
    After running the script, MLflow will store tracking data in an `mlruns` directory (created in the directory where you ran the script, typically the project root).
    To view the UI:
    * Open a new terminal in the project root directory (where `mlruns` is).
    * Run the command:
        ```bash
        mlflow ui
        ```
    * Open your web browser and go to `http://localhost:5000` (or the URL shown in the terminal).
## Models Implemented
The project trains and evaluates the following models:
* Random Forest
* Support Vector Machine (SVM)
* Multi-layer Perceptron (MLP)
* XGBoost
Configuration for these models can be found in `src/config/model_config.py`.
## Key Files
* `src/main_mlflow.py`: Main script to execute the training and evaluation pipeline with MLflow tracking.
* `src/config/model_config.py`: Contains model hyperparameters and general run settings.
* `src/preprocessing/load_data.py`: Handles loading of NFL data.
* `src/preprocessing/prepare_data.py`: Contains functions for data splitting and feature scaling.
* `src/evaluation/`: Modules for calculating metrics and plotting results.
* `src/models/`: Modules defining and training each specific model type.