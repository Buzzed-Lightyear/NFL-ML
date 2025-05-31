import xgboost as xgb
import pandas as pd

def create_xgb_model(params: dict) -> xgb.XGBClassifier:
    """
    Create an XGBoost model instance.
    
    Args:
        params: Dictionary of model parameters
        
    Returns:
        XGBClassifier: Unfitted model instance
    """
    return xgb.XGBClassifier(**params)

def train_model(model: xgb.XGBClassifier, X_train: pd.DataFrame, y_train) -> xgb.XGBClassifier:
    """
    Train the XGBoost model.
    
    Args:
        model: XGBClassifier instance
        X_train: Training features as a Pandas DataFrame
        y_train: Training labels
        
    Returns:
        XGBClassifier: Fitted model
    """
    model.fit(X_train, y_train)
    return model 