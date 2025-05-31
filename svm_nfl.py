import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.preprocessing import StandardScaler # For feature scaling
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # For coefficient analysis if using linear kernel
from util.load_nfl_year_data import load_nfl_data

# --- 1. Load Data ---
df_combined = load_nfl_data()

if not df_combined.empty and 'team1_win' in df_combined.columns:
    print("\n--- Starting SVM Model Implementation with Combined Data ---")

    # --- 2. Prepare Features (X) and Target (y) ---
    X = df_combined.drop('team1_win', axis=1)
    y = df_combined['team1_win']
    feature_names = X.columns # Save feature names for later

    print("--- Features (X) and Target (y) separated ---")
    print(f"Shape of Features (X): {X.shape}")
    print(f"Shape of Target (y): {y.shape}\n")

    # --- 3. Split Data into Training and Testing sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("--- Data split into Training and Testing sets ---")
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}\n")

    # --- 4. Feature Scaling (Crucial for SVM) ---
    print("--- Scaling features using StandardScaler ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # Fit on training data and transform it
    X_test_scaled = scaler.transform(X_test)     # Transform test data using the same scaler

    # Convert scaled arrays back to DataFrames (optional)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    print("--- Feature scaling complete! ---\n")

    # --- 5. Instantiate and Train the SVM Model ---
    # Start with a linear kernel, as suggested by the paper's coefficient analysis.
    # C is the regularization parameter. Default is 1.0.
    # probability=True allows using predict_proba, but can slow down training.
    # Not needed for basic classification metrics but useful for ROC AUC.
    svm_model = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    print(f"--- SVM Model Instantiated: {svm_model.get_params()} ---\n")

    print("--- Training the SVM model (this may take a moment)... ---")
    svm_model.fit(X_train_scaled, y_train)
    print("--- SVM Model training complete! ---\n")

    # --- 6. Make Predictions ---
    y_pred_svm = svm_model.predict(X_test_scaled)
    print("--- Predictions made on the scaled test set using SVM ---")

    # --- 7. Evaluate the SVM Model ---
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm, zero_division=0)
    recall_svm = recall_score(y_test, y_pred_svm, zero_division=0)
    f1_svm = f1_score(y_test, y_pred_svm, zero_division=0)
    cm_svm = confusion_matrix(y_test, y_pred_svm)

    print("\n--- SVM Model Evaluation Metrics: ---")
    print(f"Accuracy: {accuracy_svm:.4f}")
    print(f"Precision: {precision_svm:.4f}")
    print(f"Recall: {recall_svm:.4f}")
    print(f"F1-score: {f1_svm:.4f}")

    print("\nConfusion Matrix (SVM):")
    # Using seaborn for a nicer plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Predicted 0 (Loss/Tie)', 'Predicted 1 (Win)'],
                yticklabels=['Actual 0 (Loss/Tie)', 'Actual 1 (Win)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for SVM Model (Combined 2021-2022 Data)')
    plt.show() 

    # --- 8. Coefficient Analysis (for Linear SVM) ---
    if svm_model.kernel == 'linear':
        print("\n--- SVM Coefficients (Feature Importances for Linear SVM) ---")
        # Coefficients are only directly available for linear kernels
        coefficients = svm_model.coef_[0] # svm_model.coef_ is a 2D array for binary classification
        
        # Ensure feature_names has the correct features if VIF reduction was done prior to this script
        # Assumes 'feature_names' corresponds to columns in X_train_scaled / X_test_scaled
        coeff_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
        coeff_df['abs_coefficient'] = np.abs(coeff_df['coefficient']) # Use absolute values for ranking magnitude
        coeff_df = coeff_df.sort_values(by='abs_coefficient', ascending=False)

        print("Top 10 most impactful features (SVM Coefficients):")
        print(coeff_df.head(10))

        plt.figure(figsize=(10, 12)) # Adjust size as needed
        sns.barplot(x='coefficient', y='feature', data=coeff_df.head(30), palette="coolwarm") 
        plt.title('Feature Coefficients from Linear SVM Model (Top 30)')
        plt.xlabel('Coefficient Value (Impact on Prediction)')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()  
    else:
        print(f"\nCoefficient analysis is straightforward for linear SVM. Current kernel: {svm_model.kernel}")

else:
    if df_combined.empty:
        print("\nCannot proceed with SVM model training as the combined dataframe is empty.")
    elif 'team1_win' not in df_combined.columns:
        print("\nERROR: Target variable 'team1_win' not found in the combined DataFrame.")