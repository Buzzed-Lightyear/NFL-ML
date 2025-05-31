import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from util.load_nfl_year_data import load_nfl_data

# --- 1. Load Data ---
df_combined = load_nfl_data()


# --- 2. Proceed with the RFC Model using df_combined ---
if not df_combined.empty and 'team1_win' in df_combined.columns:
    print("\n--- Starting Model Implementation with Combined Data ---")

    X = df_combined.drop('team1_win', axis=1)
    y = df_combined['team1_win']

    print("--- Features (X) and Target (y) separated from combined data ---")
    print(f"Shape of Features (X): {X.shape}")
    print(f"Shape of Target (y): {y.shape}\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("--- Data split into Training and Testing sets ---")
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}\n")

    rfc_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    print(f"--- RFC Model Instantiated: {rfc_model.get_params()} ---\n")

    print("--- Training the model (this may take slightly longer with more data)... ---")
    rfc_model.fit(X_train, y_train)
    print("--- Model training complete! ---\n")

    y_pred = rfc_model.predict(X_test)

    print("--- Predictions made on the test set ---")

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("--- Model Evaluation Metrics (Combined Data): ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix (Combined Data):")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0 (Loss/Tie)', 'Predicted 1 (Win)'],
                yticklabels=['Actual 0 (Loss/Tie)', 'Actual 1 (Win)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for RFC Model (Combined 2021-2022 Data)')
    plt.show()

    print("\n--- Feature Importances (Combined Data) ---")
    importances = rfc_model.feature_importances_
    feature_names = X.columns # Should be the same as df_combined.drop('team1_win', axis=1).columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print("Top 10 most important features (Combined Data):")
    print(feature_importance_df.head(10))

    plt.figure(figsize=(10, 12))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(30)) # Show top 30 or all
    plt.title('Feature Importances from RFC Model (Combined 2021-2022 Data, Top 30)')
    plt.tight_layout()
    plt.show()
else:
    if df_combined.empty:
        print("\nCannot proceed with model training as the combined dataframe is empty.")
    elif 'team1_win' not in df_combined.columns:
        print("\nERROR: Target variable 'team1_win' not found in the combined DataFrame.")