from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def create_rf_model(params: dict) -> RandomForestClassifier:
    """
    Create a Random Forest model instance.
    
    Args:
        params: Dictionary of model parameters
        
    Returns:
        RandomForestClassifier: Unfitted model instance
    """
    return RandomForestClassifier(**params)

def train_model(model: RandomForestClassifier, X_train: pd.DataFrame, y_train) -> RandomForestClassifier:
    """
    Train the Random Forest model.
    
    Args:
        model: RandomForestClassifier instance
        X_train: Training features as a Pandas DataFrame
        y_train: Training labels
        
    Returns:
        RandomForestClassifier: Fitted model
    """
    model.fit(X_train, y_train)
    return model 