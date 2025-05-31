import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from util.load_nfl_year_data import load_nfl_data

# --- 1. Load Data ---
df_combined = load_nfl_data()

# --- 2. Proceed with the MLP Model using df_combined ---
if not df_combined.empty and 'team1_win' in df_combined.columns:
    print("\n--- Starting MLP Model Implementation with Combined Data ---")

    X = df_combined.drop('team1_win', axis=1)
    y = df_combined['team1_win']

    print("--- Features (X) and Target (y) separated from combined data ---")
    print(f"Shape of Features (X): {X.shape}")
    print(f"Shape of Target (y): {y.shape}\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("--- Data split into Training and Testing sets ---")
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}\n")

    # --- MLPClassifier Instantiation ---
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100, ),  # one hidden layer with 100 neurons (tweakable)
        activation='relu',           # common choice for non-linearity
        solver='adam',               # popular optimizer for neural nets
        alpha=0.0001,                # L2 penalty (regularization term)
        batch_size='auto',
        learning_rate='adaptive',    # reduce learning rate when no improvement
        max_iter=200,                # max epochs to train
        random_state=42,
        early_stopping=True,         # stop training if no improvement on validation set
        n_iter_no_change=10,
        verbose=True
    )
    print(f"--- MLP Model Instantiated: {mlp_model.get_params()} ---\n")

    print("--- Training the MLP model (this might take some time)... ---")
    mlp_model.fit(X_train, y_train)
    print("--- Model training complete! ---\n")

    y_pred = mlp_model.predict(X_test)
    y_pred_proba = mlp_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

    print("--- Predictions made on the test set ---")

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("--- Model Evaluation Metrics (MLP - Combined Data): ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix (MLP - Combined Data):")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0 (Loss/Tie)', 'Predicted 1 (Win)'],
                yticklabels=['Actual 0 (Loss/Tie)', 'Actual 1 (Win)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for MLP Model (Combined 2021-2023 Data)')
    plt.show()

    # ROC Curve for MLP
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for MLP Model')
    plt.legend(loc="lower right")
    plt.show()

else:
    if df_combined.empty:
        print("\nCannot proceed with model training as the combined dataframe is empty.")
    elif 'team1_win' not in df_combined.columns:
        print("\nERROR: Target variable 'team1_win' not found in the combined DataFrame.")
