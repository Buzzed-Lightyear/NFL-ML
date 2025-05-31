from sklearn.neural_network import MLPClassifier
import pandas as pd

def create_mlp_model(params: dict) -> MLPClassifier:
    """
    Create an MLP model instance.
    
    Args:
        params: Dictionary of model parameters
        
    Returns:
        MLPClassifier: Unfitted model instance
    """
    return MLPClassifier(**params)

def train_model(model: MLPClassifier, X_train: pd.DataFrame, y_train) -> MLPClassifier:
    """
    Train the MLP model.
    
    Args:
        model: MLPClassifier instance
        X_train: Training features as a Pandas DataFrame
        y_train: Training labels
        
    Returns:
        MLPClassifier: Fitted model
    """
    model.fit(X_train, y_train)
    return model 