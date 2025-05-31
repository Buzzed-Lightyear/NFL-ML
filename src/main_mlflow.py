import os
import argparse
import tempfile
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib.pyplot as plt
from datetime import datetime
from mlflow.models.signature import infer_signature

# --- Your existing imports ---
from preprocessing.load_data import load_nfl_data
from preprocessing.prepare_data import split_and_prepare_data, scale_features
from models.random_forest_model import create_rf_model, train_model as train_rf
from models.svm_model import create_svm_model, train_model as train_svm
from models.mlp_model import create_mlp_model, train_model as train_mlp
from models.xgboost_model import create_xgb_model, train_model as train_xgb
from evaluation.metrics_calculator import calculate_classification_metrics
from evaluation.plotting import (
    plot_confusion_matrix_heatmap,
    plot_roc_auc_curve,
    plot_model_feature_importances
)
from config.model_config import MODEL_PARAMS, TEST_SIZE, RANDOM_STATE, TARGET_COLUMN

# --- MODIFIED save_plot_to_temp ---
# This function now creates a figure and an Axes object, and passes the Axes
# to your plotting functions. Your plotting functions in src/evaluation/plotting.py
# will need to be adapted to accept and use this 'ax' argument.
def save_plot_to_temp(plot_func, *args, **kwargs):
    """
    Creates a figure and an Axes, calls the plot_func to draw on these Axes,
    saves the plot to a temporary file, and returns the path.
    The plot_func is expected to accept an 'ax' keyword argument.
    """
    fig, ax = plt.subplots(figsize=(10, 8))  # Create figure and an axes object

    # Pass the created 'ax' to the specific plotting function.
    # Ensure your functions in src/evaluation/plotting.py are modified to accept 'ax'.
    # Example: plot_confusion_matrix_heatmap(y_test, y_pred, model_name, ax=ax)
    plot_func(*args, ax=ax, **kwargs)  # Pass through other args and kwargs

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        fig.savefig(tmp.name)
        plt.close(fig)  # Close the specific figure to free memory and prevent display
        return tmp.name

def ensure_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert integer columns to float to handle potential NaN values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with integer columns converted to float
    """
    df = df.copy()
    int_cols = df.select_dtypes(include=['int', 'int32', 'int64']).columns
    if not int_cols.empty:
        print(f"Converting integer columns to float: {list(int_cols)}")
        df[int_cols] = df[int_cols].astype(float)
    return df

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate NFL prediction models with MLflow tracking')
    parser.add_argument('--experiment-name', type=str,
                        default='NFL_Game_Outcome_Prediction',
                        help='Name of the MLflow experiment')
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        mlflow.set_experiment(args.experiment_name)

        with mlflow.start_run(run_name=f"Full_Pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as parent_run:
            mlflow.log_params({
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "target_column": TARGET_COLUMN
            })

            print("Loading NFL data...")
            df_combined = load_nfl_data()
            mlflow.log_param("dataset_shape", str(df_combined.shape))

            print("\nPreparing data...")
            X_train, X_test, y_train, y_test, feature_names = split_and_prepare_data(
                df_combined,
                target_column=TARGET_COLUMN,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )

            # Convert to DataFrames and ensure float columns
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            X_train_df = ensure_float_columns(X_train_df)
            X_test_df = ensure_float_columns(X_test_df)

            results = {}
            model_functions = {
                'Random Forest': (create_rf_model, train_rf),
                'SVM': (create_svm_model, train_svm),
                'MLP': (create_mlp_model, train_mlp),
                'XGBoost': (create_xgb_model, train_xgb)
            }

            for model_name, (create_func, train_func) in model_functions.items():
                print(f"\n--- Training {model_name} Model ---")

                with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                      nested=True) as child_run:
                    mlflow.log_params(MODEL_PARAMS[model_name])

                    model = create_func(MODEL_PARAMS[model_name])

                    if model_name == 'SVM':
                        # Scale features while preserving DataFrame structure
                        X_train_scaled, X_test_scaled, _ = scale_features(X_train_df.values, X_test_df.values)
                        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
                        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
                        
                        model = train_func(model, X_train_scaled_df, y_train)
                        y_pred = model.predict(X_test_scaled_df)
                        y_pred_proba = model.predict_proba(X_test_scaled_df)[:, 1]
                        
                        input_example_df = X_train_scaled_df.head()
                        signature_input_data = X_train_scaled_df.head()
                        predictions_for_signature = model.predict(X_test_scaled_df.head())
                    else:
                        model = train_func(model, X_train_df, y_train)
                        y_pred = model.predict(X_test_df)
                        y_pred_proba = model.predict_proba(X_test_df)[:, 1]
                        
                        input_example_df = X_train_df.head()
                        signature_input_data = X_train_df.head()
                        predictions_for_signature = model.predict(X_test_df.head())

                    metrics = calculate_classification_metrics(y_test, y_pred)
                    mlflow.log_metrics(metrics)
                    results[model_name] = metrics

                    print(f"\n--- {model_name} Model Evaluation Metrics: ---")
                    for metric_name, value in metrics.items():
                        print(f"{metric_name.capitalize()}: {value:.4f}")

                    # --- Infer model signature ---
                    signature = infer_signature(signature_input_data, predictions_for_signature)

                    # --- Save and log plots ---
                    # IMPORTANT: Ensure your plotting functions in src/evaluation/plotting.py
                    # are modified to accept 'ax' as a keyword argument and draw on it.
                    # They should NOT call plt.figure(), plt.show(), or plt.close().
                    cm_path = save_plot_to_temp(plot_confusion_matrix_heatmap, y_test, y_pred, model_name)
                    mlflow.log_artifact(cm_path, "plots")
                    os.unlink(cm_path)

                    roc_path = save_plot_to_temp(plot_roc_auc_curve, y_test, y_pred_proba, model_name)
                    mlflow.log_artifact(roc_path, "plots")
                    os.unlink(roc_path)

                    fi_path = save_plot_to_temp(plot_model_feature_importances, model, feature_names, model_name)
                    mlflow.log_artifact(fi_path, "plots")
                    os.unlink(fi_path)

                    # --- Log the model WITH signature and input example ---
                    if model_name == 'XGBoost':
                        mlflow.xgboost.log_model(
                            xgb_model=model,
                            artifact_path="model",
                            signature=signature,
                            input_example=input_example_df
                        )
                    else: # For RF, SVM, MLP
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="model",
                            signature=signature,
                            input_example=input_example_df
                        )

                    # --- Save and log run configuration for this child run ---
                    run_config_payload = {
                        "model_name": model_name,
                        "model_params": MODEL_PARAMS[model_name],
                        "metrics_from_run": metrics
                    }
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_config_file:
                        json.dump(run_config_payload, tmp_config_file, indent=2)
                        config_file_path = tmp_config_file.name
                    mlflow.log_artifact(config_file_path, "config")
                    os.unlink(config_file_path)

            # --- After the loop, for the PARENT run ---
            print("\n--- Final Model Comparison (Logging to Parent Run) ---")
            comparison_df = pd.DataFrame(results).T
            comparison_df.index.name = 'model_name'
            print(comparison_df)

            with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp_comparison_file:
                comparison_df.to_csv(tmp_comparison_file.name)
                comparison_file_path = tmp_comparison_file.name
            mlflow.log_artifact(comparison_file_path, "comparison_results")
            os.unlink(comparison_file_path)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()