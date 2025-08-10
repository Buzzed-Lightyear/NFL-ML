"""Plotting utilities for model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_confusion_matrix_heatmap(y_test, y_pred, model_name: str, ax=None):
    """Plot a confusion matrix heatmap."""

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_plot = True

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0 (Loss/Tie)", "Predicted 1 (Win)"],
        yticklabels=["Actual 0 (Loss/Tie)", "Actual 1 (Win)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix for {model_name} Model")

    if show_plot:
        plt.show()


def plot_roc_auc_curve(y_test, y_pred_proba, model_name: str, ax=None):
    """Plot a ROC AUC curve."""

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_plot = True

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve for {model_name} Model")
    ax.legend(loc="lower right")

    if show_plot:
        plt.show()


def plot_model_feature_importances(model, feature_names: list, model_name: str, ax=None):
    """Plot feature importances for a fitted model."""

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 12))
        show_plot = True

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef_data = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        importances = np.abs(coef_data)
    else:
        if show_plot:
            plt.close(fig if "fig" in locals() else plt.gcf())
        print(
            f"Model {model_name} does not have feature_importances_ or coef_ attribute."
        )
        return

    if len(feature_names) != len(importances):
        if show_plot:
            plt.close(fig if "fig" in locals() else plt.gcf())
        print(
            f"Warning: Number of feature names ({len(feature_names)}) does not match number of importances ({len(importances)}) for model {model_name}."
        )
        return

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False)

    print(f"\nTop 10 most important features ({model_name}):")
    print(feature_importance_df.head(10))

    sns.barplot(
        x="importance",
        y="feature",
        data=feature_importance_df.head(30),
        ax=ax,
    )
    ax.set_title(f"Feature Importances from {model_name} Model (Top 30)")

    if "fig" in locals() and fig is not None:
        fig.tight_layout()
    elif ax.figure is not None:
        ax.figure.tight_layout()

    if show_plot:
        plt.show()


def plot_reliability_curve(
    y_true,
    y_prob,
    n_bins: int = 10,
    title: str | None = None,
    out_path: str | None = None,
):
    """Render a reliability diagram (observed vs predicted).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int, default 10
        Number of bins to use.
    title : str, optional
        Optional plot title.
    out_path : str, optional
        If provided, the plot is saved to this path.
    """

    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(y_prob, 0.0, 1.0 - 1e-12)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges, right=True)
    bin_ids = np.minimum(bin_ids, n_bins)

    obs = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=int)
    conf = np.full(n_bins, np.nan)

    for i in range(1, n_bins + 1):
        m = bin_ids == i
        if m.any():
            obs[i - 1] = y_true[m].mean()
            conf[i - 1] = y_prob[m].mean()
            cnt[i - 1] = m.sum()

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(conf, obs, marker="o")
    sizes = 50 * (cnt / cnt.max() if cnt.max() > 0 else 1)
    plt.scatter(conf, obs, s=sizes)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title or f"Reliability diagram (n_bins={n_bins})")
    plt.grid(True)

    if out_path:
        import pathlib

        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)

    return plt.gcf()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a reliability diagram from a predictions CSV."
    )
    parser.add_argument(
        "csv_path", help="Path to CSV with columns 'y_true' and 'y_pred_proba'"
    )
    parser.add_argument(
        "--out",
        default="reports/reliability.png",
        help="Output path for the PNG diagram",
    )
    parser.add_argument("--n-bins", type=int, default=10, dest="n_bins")
    parser.add_argument("--title", default=None, help="Optional plot title")

    cli_args = parser.parse_args()
    df_preds = pd.read_csv(cli_args.csv_path)
    plot_reliability_curve(
        df_preds["y_true"],
        df_preds["y_pred_proba"],
        n_bins=cli_args.n_bins,
        title=cli_args.title,
        out_path=cli_args.out,
    )

