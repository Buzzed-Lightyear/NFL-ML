import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV # Import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from util.load_nfl_year_data import load_nfl_data # Assuming this is your data loader

# --- 1. Load Data ---
df_combined = load_nfl_data() # This should load your 3-season data

# --- 2. Proceed with the RFC Model using df_combined ---
if not df_combined.empty and 'team1_win' in df_combined.columns:
    print("\n--- Starting Model Implementation with Combined Data (with GridSearchCV for RFC) ---")

    X = df_combined.drop('team1_win', axis=1)
    y = df_combined['team1_win']

    print("--- Features (X) and Target (y) separated from combined data ---")
    print(f"Shape of Features (X): {X.shape}") # Should be (852, 54) with 3 seasons
    print(f"Shape of Target (y): {y.shape}\n")   # Should be (852,)

    # Use the same train_test_split as before to ensure comparability if needed
    # but GridSearchCV will perform its own splits on X_train_full, y_train_full for CV
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("--- Data split into Training (for GridSearchCV) and Testing sets ---")
    print(f"Shape of X_train_full (for GridSearchCV): {X_train_full.shape}") # (681, 54)
    print(f"Shape of X_test: {X_test.shape}\n")                     # (171, 54)

    # --- Define Parameter Grid for RFC ---
    # param_grid_rfc = {
    #     'n_estimators': [100, 200, 300],       # Reduced for a slightly faster example, expand as desired
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2],
    #     'max_features': ['sqrt', 'log2']
    #     # 'bootstrap': [True] # Keep bootstrap True for now unless specific reason to change
    # }
    # Current example grid: 3 * 2 * 3 * 2 * 2 * 2 = 144 combinations. With cv=5, that's 720 fits.
    # Your more exhaustive grid (864 combinations) would be:
    param_grid_rfc = {
       'n_estimators': [100, 200, 300, 400],
       'criterion': ['gini', 'entropy'],
       'max_depth': [None, 10, 20, 30],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 2, 4],
       'max_features': ['sqrt', 'log2', None]
    }


    print(f"--- RFC Parameter Grid Defined for GridSearchCV ---")
    print(param_grid_rfc)

    # --- Instantiate and Run GridSearchCV ---
    # Using 'accuracy' as the scoring metric. You could also use 'f1_weighted' or other relevant metrics.
    # cv=5 means 5-fold cross-validation.
    # n_jobs=-1 will use all available CPU cores to speed up the search.
    # verbose=2 will give more detailed output during the search process.
    rfc_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                                   param_grid=param_grid_rfc,
                                   cv=5,
                                   scoring='accuracy',
                                   verbose=2, # Increased verbosity
                                   n_jobs=-1)

    print("\n--- Fitting GridSearchCV for RFC (this might take a very long time depending on the grid size)... ---")
    # Fit GridSearchCV on the full training set (it handles its own internal CV splits)
    # RFC doesn't strictly require scaled data, so using X_train_full directly
    rfc_grid_search.fit(X_train_full, y_train_full)
    print("--- GridSearchCV for RFC complete! ---")

    # --- Get Best Parameters and Score ---
    print(f"\nBest RFC parameters found: {rfc_grid_search.best_params_}")
    print(f"Best cross-validation accuracy score from GridSearchCV: {rfc_grid_search.best_score_:.4f}")

    # --- Evaluate the Best RFC Model (found by GridSearchCV) on the Test Set ---
    best_rfc_model = rfc_grid_search.best_estimator_ # This is the model with the best found parameters, already refitted on the whole X_train_full

    print("\n--- Evaluating Best RFC Model on the Test Set ---")
    y_pred_best_rfc = best_rfc_model.predict(X_test)

    accuracy_best = accuracy_score(y_test, y_pred_best_rfc)
    precision_best = precision_score(y_test, y_pred_best_rfc, zero_division=0)
    recall_best = recall_score(y_test, y_pred_best_rfc, zero_division=0)
    f1_best = f1_score(y_test, y_pred_best_rfc, zero_division=0)
    cm_best = confusion_matrix(y_test, y_pred_best_rfc)

    print("--- Best RFC Model Evaluation Metrics (Test Set): ---")
    print(f"Accuracy: {accuracy_best:.4f}")
    print(f"Precision: {precision_best:.4f}")
    print(f"Recall: {recall_best:.4f}")
    print(f"F1-score: {f1_best:.4f}")
    print("\nConfusion Matrix for Best RFC Model (Test Set):")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0 (Loss/Tie)', 'Predicted 1 (Win)'],
                yticklabels=['Actual 0 (Loss/Tie)', 'Actual 1 (Win)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Tuned RFC Model ({len(df_combined)/len(X.columns):.0f} Seasons Data)') # Dynamic title
    plt.show()

    print("\n--- Feature Importances from Best Tuned RFC Model ---")
    importances_best = best_rfc_model.feature_importances_
    feature_names = X.columns
    feature_importance_df_best = pd.DataFrame({'feature': feature_names, 'importance': importances_best})
    feature_importance_df_best = feature_importance_df_best.sort_values(by='importance', ascending=False)

    print("Top 10 most important features (Best Tuned RFC Model):")
    print(feature_importance_df_best.head(10))

    plt.figure(figsize=(10, 12)) # Adjusted for potentially many features
    # Display more features if the list is long, or adjust head()
    num_features_to_plot = min(30, len(feature_names))
    sns.barplot(x='importance', y='feature', data=feature_importance_df_best.head(num_features_to_plot))
    plt.title(f'Feature Importances from Tuned RFC Model (Top {num_features_to_plot})')
    plt.tight_layout()
    plt.show()

else:
    if df_combined.empty:
        print("\nCannot proceed with model training as the combined dataframe is empty.")
    elif 'team1_win' not in df_combined.columns:
        print("\nERROR: Target variable 'team1_win' not found in the combined DataFrame.")