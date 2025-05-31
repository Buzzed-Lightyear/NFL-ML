from sklearn.svm import SVC
import pandas as pd

def create_svm_model(params: dict) -> SVC:
    """
    Create an SVM model instance.
    
    Args:
        params: Dictionary of model parameters
        
    Returns:
        SVC: Unfitted model instance
    """
    return SVC(**params)

def train_model(model: SVC, X_train: pd.DataFrame, y_train) -> SVC:
    """
    Train the SVM model.
    
    Args:
        model: SVC instance
        X_train: Training features as a Pandas DataFrame (should be scaled)
        y_train: Training labels
        
    Returns:
        SVC: Fitted model
    """
    model.fit(X_train, y_train)
    return model 