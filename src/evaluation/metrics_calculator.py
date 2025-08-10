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

def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Compute the expected calibration error (ECE)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges, right=True)
    ece = 0.0
    for i in range(1, n_bins + 1):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        prob_mean = y_prob[mask].mean()
        label_mean = y_true[mask].mean()
        ece += mask.mean() * abs(prob_mean - label_mean)
    return float(ece)

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
        "expected_calibration_error": expected_calibration_error(y_true, y_proba),
    }
    return metrics