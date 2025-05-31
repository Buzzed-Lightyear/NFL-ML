from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_classification_metrics(y_test, y_pred):
    """
    Calculate classification metrics for model evaluation.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1 score
    """
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    return metrics 