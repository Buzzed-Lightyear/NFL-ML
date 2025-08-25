import os
import argparse
import tempfile
import json
import hashlib
import subprocess
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib.pyplot as plt
from datetime import datetime
from mlflow.models.signature import infer_signature

from preprocessing.load_data import load_nfl_data
from preprocessing.prepare_data import scale_features
from models.registry import get_model
from evaluation.metrics_calculator import calculate_classification_metrics
from evaluation.plotting import (
    plot_confusion_matrix_heatmap,
    plot_roc_auc_curve,
    plot_model_feature_importances
)
from config.model_config import (
    DATA_YEARS,
    MODELS_TO_TRAIN,
    MODEL_PARAMS,
    TARGET_COLUMN,
)

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
    parser.add_argument('--train-years', type=str, default=None,
                        help='Comma-separated list of training years')
    parser.add_argument('--eval-years', type=str, default=None,
                        help='Comma-separated list of evaluation years')
    parser.add_argument('--test-years', type=str, default=None,
                        help='Comma-separated list of test years')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated list of models to train')
    parser.add_argument('--run-type', type=str, default='baseline',
                        help='Tag describing the run type')
    parser.add_argument('--split-id', type=str, default=None,
                        help='Optional predefined split identifier')
    parser.add_argument('--data-spec', type=str, default=None,
                        help='Optional path to YAML dataset spec; if provided, load data via spec.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Optional output base dir for processed artifacts; defaults to data/processed/<run_id>.')
    parser.add_argument(
        '--ece-bins',
        type=int,
        default=10,
        help=(
            'Number of bins for additional ECE/MCE metrics (2-100). '
            'Standard metrics with 10 and 15 bins are always included.'
        ),
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.ece_bins < 2 or args.ece_bins > 100:
        raise ValueError(
            f"--ece-bins must be between 2 and 100, got {args.ece_bins}"
        )

    try:
        mlflow.set_experiment(args.experiment_name)

        train_years = list(map(int, args.train_years.split(','))) if args.train_years else DATA_YEARS['train']
        eval_years = list(map(int, args.eval_years.split(','))) if args.eval_years else DATA_YEARS['eval']
        test_years = list(map(int, args.test_years.split(','))) if args.test_years else DATA_YEARS['test']
        models_to_train = args.models.split(',') if args.models else MODELS_TO_TRAIN

        with mlflow.start_run(run_name=f"Full_Pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as parent_run:
            run_id = parent_run.info.run_id

            split_descriptor = json.dumps({'train': train_years, 'eval': eval_years, 'test': test_years}, sort_keys=True)
            split_id = args.split_id or hashlib.md5(split_descriptor.encode()).hexdigest()[:8]

            os.makedirs('data/splits', exist_ok=True)
            with open(f'data/splits/{run_id}.json', 'w') as f:
                json.dump({'train': train_years, 'eval': eval_years, 'test': test_years}, f, indent=2)

            print("Loading NFL data...")
            if args.data_spec:
                try:
                    from preprocessing.unified_loader import load_dataset_from_spec
                    df_combined, data_meta = load_dataset_from_spec(args.data_spec)
                except Exception as e:
                    raise RuntimeError(f"Failed to load spec dataset: {e}") from e
                if df_combined.empty:
                    raise ValueError("Spec-loaded dataset is empty")
                if 'season' not in df_combined.columns:
                    raise ValueError("Spec-loaded dataset must include 'season' column")
                print("Using data source: spec")
                print(f"Spec name: {data_meta.get('dataset_name')}")
                print(f"data_grain: {data_meta.get('data_grain')}")
                print(f"modeling_grain: {data_meta.get('modeling_grain')}")
                print(f"input_fingerprint: {data_meta.get('input_fingerprint')}")
                mlflow.log_params({
                    "data_spec_path": args.data_spec,
                    "data_spec_name": data_meta.get("dataset_name"),
                    "data_grain": data_meta.get("data_grain"),
                    "modeling_grain": data_meta.get("modeling_grain"),
                    "input_fingerprint": data_meta.get("input_fingerprint"),
                })
                data_source_tag = 'spec'
                spec_name = data_meta.get("dataset_name")
            else:
                df_combined = load_nfl_data(years=train_years + eval_years + test_years)
                if df_combined.empty:
                    raise ValueError("Loaded dataset is empty")
                data_meta = None
                data_source_tag = 'legacy'
                spec_name = None
                print("Using data source: legacy")

            mlflow.log_param("dataset_shape", str(df_combined.shape))

            train_df = df_combined[df_combined['season'].isin(train_years)].copy()
            eval_df = df_combined[df_combined['season'].isin(eval_years)].copy()
            test_df = df_combined[df_combined['season'].isin(test_years)].copy()

            metadata_cols = ['season', 'week', 'kickoff_ts', 'home_team', 'away_team']
            feature_cols = [c for c in train_df.columns if c not in metadata_cols + [TARGET_COLUMN]]

            feature_hash = hashlib.md5(','.join(sorted(feature_cols)).encode()).hexdigest()[:8]
            mlflow.log_param("feature_count", len(feature_cols))
            mlflow.log_param("feature_hash", feature_hash)
            data_snapshot_ts = datetime.utcnow().isoformat()
            code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

            def enrich(df: pd.DataFrame) -> pd.DataFrame:
                df = df.copy()
                for col in metadata_cols:
                    if col not in df.columns:
                        df[col] = pd.NA
                df['feature_config_hash'] = feature_hash
                df['data_snapshot_ts'] = data_snapshot_ts
                df['code_commit'] = code_commit
                df['split_id'] = split_id
                return df

            train_df_meta = enrich(train_df)
            eval_df_meta = enrich(eval_df)
            test_df_meta = enrich(test_df)

            if args.output_dir:
                out_dir = Path(args.output_dir) / run_id
            else:
                out_dir = Path('data/processed') / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Resolved output directory: {out_dir}")
            train_df_meta.to_parquet(out_dir / 'train.parquet', index=False)
            eval_df_meta.to_parquet(out_dir / 'eval.parquet', index=False)
            test_df_meta.to_parquet(out_dir / 'test.parquet', index=False)

            if args.data_spec:
                mlflow.log_artifact(args.data_spec, artifact_path='specs')
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_meta_file:
                    json.dump(data_meta, tmp_meta_file, indent=2)
                    meta_path = tmp_meta_file.name
                mlflow.log_artifact(meta_path, artifact_path='specs')
                os.unlink(meta_path)
                mlflow.log_artifact(out_dir / 'train.parquet', artifact_path='data')
                mlflow.log_artifact(out_dir / 'eval.parquet', artifact_path='data')
                mlflow.log_artifact(out_dir / 'test.parquet', artifact_path='data')

            X_train = ensure_float_columns(train_df_meta[feature_cols])
            y_train = train_df_meta[TARGET_COLUMN]
            X_eval = ensure_float_columns(eval_df_meta[feature_cols])
            y_eval = eval_df_meta[TARGET_COLUMN]

            tags = {
                'run_type': args.run_type,
                'split_id': split_id,
                'feature_hash': feature_hash,
                'years_train': ','.join(map(str, train_years)),
                'years_eval': ','.join(map(str, eval_years)),
                'models': ','.join(models_to_train),
                'data_source': data_source_tag,
            }
            if spec_name:
                tags['spec_name'] = spec_name
            mlflow.set_tags(tags)

            results = {}
            model_name_map = {'rf': 'Random Forest', 'svm': 'SVM', 'mlp': 'MLP', 'xgb': 'XGBoost'}

            for model_key in models_to_train:
                readable_name = model_name_map.get(model_key, model_key)
                print(f"\n--- Training {readable_name} Model ---")

                model, train_func = get_model(model_key)

                with mlflow.start_run(
                    run_name=f"{readable_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    nested=True,
                ) as child_run:
                    mlflow.set_tags({'model': model_key, 'split_id': split_id, 'feature_hash': feature_hash, 'run_type': args.run_type})
                    mlflow.log_params(MODEL_PARAMS[model_key])
                    if model_key == 'svm':
                        X_train_scaled, X_eval_scaled, _ = scale_features(X_train.values, X_eval.values)
                        X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
                        X_eval_df = pd.DataFrame(X_eval_scaled, columns=feature_cols)
                        model = train_func(model, X_train_df, y_train)
                        y_pred = model.predict(X_eval_df)
                        y_pred_proba = model.predict_proba(X_eval_df)[:, 1]
                        input_example_df = X_train_df.head()
                        signature_input_data = X_train_df.head()
                        predictions_for_signature = model.predict(X_eval_df.head())
                    else:
                        model = train_func(model, X_train, y_train)
                        y_pred = model.predict(X_eval)
                        y_pred_proba = model.predict_proba(X_eval)[:, 1]
                        input_example_df = X_train.head()
                        signature_input_data = X_train.head()
                        predictions_for_signature = model.predict(X_eval.head())

                    extra_bins = (
                        args.ece_bins if args.ece_bins not in {10, 15} else None
                    )
                    metrics = calculate_classification_metrics(
                        y_eval, y_pred, y_pred_proba, ece_bins=extra_bins
                    )
                    mlflow.log_metrics(metrics)
                    results[readable_name] = metrics

                    print(f"\n--- {readable_name} Model Evaluation Metrics: ---")
                    for metric_name, value in metrics.items():
                        print(f"{metric_name.capitalize()}: {value:.4f}")

                    signature = infer_signature(signature_input_data, predictions_for_signature)

                    cm_path = save_plot_to_temp(plot_confusion_matrix_heatmap, y_eval, y_pred, readable_name)
                    mlflow.log_artifact(cm_path, 'plots')
                    os.unlink(cm_path)

                    roc_path = save_plot_to_temp(plot_roc_auc_curve, y_eval, y_pred_proba, readable_name)
                    mlflow.log_artifact(roc_path, 'plots')
                    os.unlink(roc_path)

                    fi_path = save_plot_to_temp(plot_model_feature_importances, model, feature_cols, readable_name)
                    mlflow.log_artifact(fi_path, 'plots')
                    os.unlink(fi_path)

                    if model_key == 'xgb':
                        mlflow.xgboost.log_model(
                            xgb_model=model,
                            artifact_path='model',
                            signature=signature,
                            input_example=input_example_df
                        )
                    else:
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path='model',
                            signature=signature,
                            input_example=input_example_df
                        )

                    run_config_payload = {
                        'model_name': readable_name,
                        'model_params': MODEL_PARAMS[model_key],
                        'metrics_from_run': metrics,
                    }
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_config_file:
                        json.dump(run_config_payload, tmp_config_file, indent=2)
                        config_file_path = tmp_config_file.name
                    mlflow.log_artifact(config_file_path, 'config')
                    os.unlink(config_file_path)

            print("\n--- Final Model Comparison (Logging to Parent Run) ---")
            comparison_df = pd.DataFrame(results).T
            comparison_df.index.name = 'model_name'
            print(comparison_df)

            with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp_comparison_file:
                comparison_df.to_csv(tmp_comparison_file.name)
                comparison_file_path = tmp_comparison_file.name
            mlflow.log_artifact(comparison_file_path, 'comparison_results')
            os.unlink(comparison_file_path)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
