import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb # Import XGBoost
from util.load_nfl_year_data import load_nfl_data

# --- 1. Load Data ---
df_combined = load_nfl_data()

# --- 2. Proceed with the XGBoost Model using df_combined ---
if not df_combined.empty and 'team1_win' in df_combined.columns:
    print("\n--- Starting XGBoost Model Implementation with Combined Data ---")

    X = df_combined.drop('team1_win', axis=1)
    y = df_combined['team1_win']

    print("--- Features (X) and Target (y) separated from combined data ---")
    print(f"Shape of Features (X): {X.shape}")
    print(f"Shape of Target (y): {y.shape}\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("--- Data split into Training and Testing sets ---")
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}\n")

    # Instantiate XGBoost Classifier
    # Common parameters for a good starting point:
    # objective='binary:logistic' for binary classification (outputs probabilities)
    # eval_metric='logloss' for evaluation during training
    # use_label_encoder=False to suppress a common warning for recent XGBoost versions
    # n_estimators: number of boosting rounds (trees)
    # learning_rate: shrinks the contribution of each tree
    # max_depth: maximum depth of a tree
    # subsample: fraction of samples used for fitting the individual base learners
    # colsample_bytree: fraction of features used for fitting the individual base learners
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False, # Suppress warning
        n_estimators=100,        # Similar to RFC's n_estimators (number of trees)
        learning_rate=0.1,       # Typical value, often tuned
        max_depth=3,             # Typical value, often tuned; smaller than RFC's None for gradient boosting
        subsample=0.8,           # Fraction of samples for each tree
        colsample_bytree=0.8,    # Fraction of features for each tree
        random_state=42,
        n_jobs=-1                # Use all available CPU cores
    )
    print(f"--- XGBoost Model Instantiated: {xgb_model.get_params()} ---\n")

    print("--- Training the XGBoost model (this may take slightly longer with more data)... ---")
    xgb_model.fit(X_train, y_train)
    print("--- Model training complete! ---\n")

    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1] # Get probabilities for ROC curve

    print("--- Predictions made on the test set ---")

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("--- Model Evaluation Metrics (XGBoost - Combined Data): ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix (XGBoost - Combined Data):")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0 (Loss/Tie)', 'Predicted 1 (Win)'],
                yticklabels=['Actual 0 (Loss/Tie)', 'Actual 1 (Win)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for XGBoost Model (Combined 2021-2023 Data)')
    plt.show()

    # ROC Curve for XGBoost
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for XGBoost Model')
    plt.legend(loc="lower right")
    plt.show()

    print("\n--- Feature Importances (XGBoost - Combined Data) ---")
    # Feature importances from XGBoost are typically available via .feature_importances_
    # which calculates the importance based on how often a feature is used in the splits across all trees.
    importances = xgb_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print("Top 10 most important features (XGBoost - Combined Data):")
    print(feature_importance_df.head(10))

    plt.figure(figsize=(10, 12))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(30)) # Show top 30 or all
    plt.title('Feature Importances from XGBoost Model (Combined 2021-2023 Data, Top 30)')
    plt.tight_layout()
    plt.show()
else:
    if df_combined.empty:
        print("\nCannot proceed with model training as the combined dataframe is empty.")
    elif 'team1_win' not in df_combined.columns:
        print("\nERROR: Target variable 'team1_win' not found in the combined DataFrame.")