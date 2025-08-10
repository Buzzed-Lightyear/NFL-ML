from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
import numpy as np

def calculate_classification_metrics(y_true, y_pred, y_proba):
    """Calculate classification metrics for model evaluation.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted class labels.
    y_proba : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'log_loss': log_loss(y_true, y_proba),
        'brier_score': brier_score_loss(y_true, y_proba),
        'calibration_error': np.mean(np.abs(y_true - y_proba)),
    }
    return metrics