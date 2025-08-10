import pandas as pd
from preprocessing.load_data import load_nfl_data
from preprocessing.prepare_data import split_and_prepare_data, scale_features
from evaluation.metrics_calculator import calculate_classification_metrics
from evaluation.plotting import (
    plot_confusion_matrix_heatmap,
    plot_roc_auc_curve,
    plot_model_feature_importances
)
from config.model_config import MODEL_PARAMS, TEST_SIZE, RANDOM_STATE, TARGET_COLUMN, MODELS_TO_TRAIN
from models.registry import get_model

def main():
    try:
        # Load data
        print("Loading NFL data...")
        df_combined = load_nfl_data()
        
        # Prepare data
        print("\nPreparing data...")
        X_train, X_test, y_train, y_test, feature_names = split_and_prepare_data(
            df_combined, 
            target_column=TARGET_COLUMN,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        
        # Dictionary to store results
        results = {}

        model_name_map = {
            'rf': 'Random Forest',
            'svm': 'SVM',
            'mlp': 'MLP',
            'xgb': 'XGBoost',
        }

        # Train and evaluate each model
        for model_key in MODELS_TO_TRAIN:
            readable_name = model_name_map.get(model_key, model_key)
            print(f"\n--- Training {readable_name} Model ---")

            model, train_func = get_model(model_key)

            # Special handling for SVM (needs scaling)
            if model_key == 'svm':
                X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
                model = train_func(model, X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model = train_func(model, X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            results[readable_name] = metrics

            # Print metrics
            print(f"\n--- {readable_name} Model Evaluation Metrics: ---")
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")

            # Generate plots
            plot_confusion_matrix_heatmap(y_test, y_pred, readable_name)
            plot_roc_auc_curve(y_test, y_pred_proba, readable_name)
            plot_model_feature_importances(model, feature_names, readable_name)

        # Print comparison of all models
        print("\n--- Model Comparison ---")
        comparison_df = pd.DataFrame(results).T
        print(comparison_df)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 