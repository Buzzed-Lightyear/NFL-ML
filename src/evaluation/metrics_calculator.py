"""Utility functions for computing classification metrics."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Compute the expected calibration error (ECE). 

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins. ``n_bins`` must be at least ``2`` and cannot exceed the
        number of samples.

    Returns
    -------
    float
        Weighted average absolute gap between confidence and accuracy.

    Examples
    --------
    >>> expected_calibration_error([0, 1], [0.2, 0.8], n_bins=2)
    0.0
    """

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0 - 1e-12)

    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    if n_bins > len(y_true):
        raise ValueError(
            f"n_bins ({n_bins}) cannot exceed number of samples ({len(y_true)})"
        )
    if n_bins > len(y_true) / 2:
        warnings.warn(
            "n_bins is high relative to the sample size; calibration estimates may be noisy",
            stacklevel=2,
        )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges, right=True)
    bin_ids = np.minimum(bin_ids, n_bins)

    ece = 0.0
    for i in range(1, n_bins + 1):
        mask = bin_ids == i
        if not mask.any():
            continue
        prob_mean = y_prob[mask].mean()
        label_mean = y_true[mask].mean()
        bin_weight = mask.sum() / y_true.size
        ece += bin_weight * abs(prob_mean - label_mean)

    return float(ece)


def maximum_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Compute the maximum calibration error (MCE). 

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins. ``n_bins`` must be at least ``2`` and cannot exceed the
        number of samples.

    Returns
    -------
    float
        Maximum absolute calibration gap across bins.

    Examples
    --------
    >>> maximum_calibration_error([0, 1], [0.2, 0.8], n_bins=2)
    0.0
    """

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0 - 1e-12)

    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    if n_bins > len(y_true):
        raise ValueError(
            f"n_bins ({n_bins}) cannot exceed number of samples ({len(y_true)})"
        )
    if n_bins > len(y_true) / 2:
        warnings.warn(
            "n_bins is high relative to the sample size; calibration estimates may be noisy",
            stacklevel=2,
        )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges, right=True)
    bin_ids = np.minimum(bin_ids, n_bins)

    mce = 0.0
    for i in range(1, n_bins + 1):
        mask = bin_ids == i
        if not mask.any():
            continue
        prob_mean = y_prob[mask].mean()
        label_mean = y_true[mask].mean()
        mce = max(mce, abs(prob_mean - label_mean))

    return float(mce)


def calculate_classification_metrics(
    y_true,
    y_pred,
    y_proba,
    ece_bins: int | None = None,
):
    """Calculate classification metrics for model evaluation.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted class labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    ece_bins : int, optional
        If provided, computes additional ``ece_{bins}`` and ``mce_{bins}``
        metrics using that number of bins. Metrics with 10 and 15 bins are
        always computed for comparability and should not be passed here.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    calibration_mae = np.mean(np.abs(y_true - y_proba))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "calibration_mae": calibration_mae,
        # TODO: remove legacy alias in a future release
        "calibration_error": calibration_mae,
        "ece_10": expected_calibration_error(y_true, y_proba, n_bins=10),
        "ece_15": expected_calibration_error(y_true, y_proba, n_bins=15),
        "mce_10": maximum_calibration_error(y_true, y_proba, n_bins=10),
    }

    if ece_bins is not None and ece_bins not in {10, 15}:
        metrics[f"ece_{ece_bins}"] = expected_calibration_error(
            y_true, y_proba, n_bins=ece_bins
        )
        metrics[f"mce_{ece_bins}"] = maximum_calibration_error(
            y_true, y_proba, n_bins=ece_bins
        )

    return metrics

