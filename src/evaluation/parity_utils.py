"""Utility helpers for parity checking between legacy and spec pipelines."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mlflow
import pandas as pd

# Metrics we care about for parity
METRICS = [
    "accuracy",
    "log_loss",
    "brier_score",
    "roc_auc",
    "ece_10",
    "ece_15",
]


def load_mlflow_metrics(run_id: str) -> Dict[str, Dict[str, float]]:
    """Return metrics for each model under a given MLflow run.

    Parameters
    ----------
    run_id: str
        The parent run identifier from ``main_mlflow.py``.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping of model key to metric dictionary.
    """
    run = mlflow.get_run(run_id)
    exp_id = run.info.experiment_id
    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
    )
    metrics: Dict[str, Dict[str, float]] = {}
    for _, row in runs.iterrows():
        model = row.get("tags.model")
        if not model:
            continue
        metrics[model] = {
            m: row.get(f"metrics.{m}") for m in METRICS if row.get(f"metrics.{m}") is not None
        }
    return metrics


def load_split_parquet(run_id: str, base_dir: str = "data/processed") -> Dict[str, pd.DataFrame]:
    """Load train/eval/test parquet files for a given run."""
    base = Path(base_dir) / run_id
    dfs: Dict[str, pd.DataFrame] = {}
    for split in ("train", "eval", "test"):
        path = base / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Expected parquet file not found: {path}")
        dfs[split] = pd.read_parquet(path)
    return dfs


def compute_feature_set(
    df: pd.DataFrame, target_col: str, metadata_cols: Iterable[str]
) -> Tuple[set, str]:
    """Compute feature column set and its hash."""
    feature_cols = [c for c in df.columns if c not in set(metadata_cols) | {target_col}]
    feature_hash = hashlib.md5(
        ",".join(sorted(feature_cols)).encode()
    ).hexdigest()[:8]
    return set(feature_cols), feature_hash


def compute_prevalence(df: pd.DataFrame, target_col: str) -> float:
    """Compute prevalence (mean of target column)."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame")
    return float(df[target_col].mean())


def compare_metrics(
    m_legacy: Dict[str, Dict[str, float]],
    m_spec: Dict[str, Dict[str, float]],
    tolerances: Dict[str, float],
) -> List[str]:
    """Compare metrics dictionaries and return list of failures."""
    failures: List[str] = []
    for model, legacy_vals in m_legacy.items():
        spec_vals = m_spec.get(model)
        if spec_vals is None:
            failures.append(f"Missing metrics for model '{model}' in spec run")
            continue
        for metric, tol in tolerances.items():
            l_val = legacy_vals.get(metric)
            s_val = spec_vals.get(metric)
            if l_val is None or s_val is None:
                continue
            if abs(l_val - s_val) > tol:
                failures.append(
                    f"{model} {metric} diff {abs(l_val - s_val):.4f} exceeds tol {tol}"
                )
    return failures


def compare_schema(
    counts: Dict[str, Tuple[int, int]],
    prevalence: Dict[str, Tuple[float, float]],
    features: Dict[str, object],
    count_tol: int = 0,
    prevalence_tol: float = 0.005,
) -> List[str]:
    """Compare row counts, prevalence and feature sets."""
    failures: List[str] = []
    for split, (c_leg, c_spec) in counts.items():
        if abs(c_leg - c_spec) > count_tol:
            failures.append(
                f"Row count mismatch for {split}: legacy={c_leg}, spec={c_spec}"
            )
    for split, (p_leg, p_spec) in prevalence.items():
        if abs(p_leg - p_spec) > prevalence_tol:
            failures.append(
                f"Prevalence mismatch for {split}: legacy={p_leg:.4f}, spec={p_spec:.4f}"
            )
    if features.get("legacy") != features.get("spec"):
        diff_legacy = sorted(features["legacy"] - features["spec"])
        diff_spec = sorted(features["spec"] - features["legacy"])
        failures.append(
            f"Feature columns differ: legacy_only={diff_legacy}, spec_only={diff_spec}"
        )
    if features.get("hash_legacy") != features.get("hash_spec"):
        failures.append(
            f"Feature hash mismatch: {features.get('hash_legacy')} vs {features.get('hash_spec')}"
        )
    return failures


def render_markdown_report(
    legacy_run_id: str,
    spec_run_id: str,
    counts: Dict[str, Dict[str, float]],
    prevalence: Dict[str, Dict[str, float]],
    feature_info: Dict[str, object],
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    tolerances: Dict[str, Dict[str, float]],
    conclusion: str,
) -> str:
    """Render a markdown parity report."""
    lines: List[str] = []
    lines.append(f"# Parity Report\n")
    lines.append(f"Legacy run: `{legacy_run_id}`  ")
    lines.append(f"Spec run: `{spec_run_id}`\n")

    lines.append("## Row Counts and Prevalence\n")
    lines.append(
        "| Split | Legacy Count | Spec Count | Δ | Legacy Prev | Spec Prev | Δ |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for split in counts:
        c = counts[split]
        p = prevalence[split]
        lines.append(
            f"| {split} | {c['legacy']} | {c['spec']} | {c['delta']} | "
            f"{p['legacy']:.4f} | {p['spec']:.4f} | {p['delta']:.4f} |"
        )

    lines.append("\n## Feature Comparison\n")
    lines.append("| | Legacy | Spec |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Feature Count | {feature_info['count_legacy']} | {feature_info['count_spec']} |"
    )
    lines.append(
        f"| Feature Hash | {feature_info['hash_legacy']} | {feature_info['hash_spec']} |"
    )
    if feature_info.get("diff_legacy_only") or feature_info.get("diff_spec_only"):
        lines.append("\nMismatched columns:")
        lines.append(
            f"- Legacy only: {sorted(feature_info.get('diff_legacy_only', []))}\n"
            f"- Spec only: {sorted(feature_info.get('diff_spec_only', []))}"
        )

    lines.append("\n## Metrics Comparison\n")
    lines.append(
        "| Model | Metric | Legacy | Spec | Δ | Tol | Pass |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for model, metric_vals in metrics.items():
        for metric, vals in metric_vals.items():
            lines.append(
                f"| {model} | {metric} | {vals['legacy']:.4f} | {vals['spec']:.4f} | "
                f"{vals['delta']:.4f} | {vals['tolerance']} | {vals['passed']} |"
            )

    lines.append("\n## Conclusion\n")
    lines.append(conclusion)
    return "\n".join(lines)


def save_report_and_json(run_id_parent: str, md_str: str, json_obj: dict) -> None:
    """Save report and summary as MLflow artifacts."""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tmp_md:
        tmp_md.write(md_str)
        md_path = tmp_md.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp_json:
        json.dump(json_obj, tmp_json, indent=2)
        json_path = tmp_json.name
    mlflow.log_artifact(md_path, artifact_path="parity")
    mlflow.log_artifact(json_path, artifact_path="parity")
    Path(md_path).unlink(missing_ok=True)
    Path(json_path).unlink(missing_ok=True)