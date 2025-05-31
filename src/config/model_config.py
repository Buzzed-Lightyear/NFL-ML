"""
Configuration settings for all models and general parameters.
"""

# General settings
TEST_SIZE = 0.20
RANDOM_STATE = 42
TARGET_COLUMN = 'team1_win'

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'random_state': RANDOM_STATE
}

SVM_PARAMS = {
    'kernel': 'linear',
    'C': 1.0,
    'random_state': RANDOM_STATE,
    'probability': True
}

MLP_PARAMS = {
    'hidden_layer_sizes': (100,),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'batch_size': 'auto',
    'learning_rate': 'adaptive',
    'max_iter': 200,
    'random_state': RANDOM_STATE,
    'early_stopping': True,
    'n_iter_no_change': 10,
    'verbose': True
}

XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Dictionary mapping model names to their parameters
MODEL_PARAMS = {
    'Random Forest': RANDOM_FOREST_PARAMS,
    'SVM': SVM_PARAMS,
    'MLP': MLP_PARAMS,
    'XGBoost': XGBOOST_PARAMS
} 