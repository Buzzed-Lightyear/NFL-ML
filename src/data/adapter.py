"""Core adapter functions for transforming raw CSVs into a modeling table.

This module implements a schema-aware data loader that reads arbitrary CSV files
based on a YAML :mod:`spec` definition. The goal for Phase A is parity with the
existing game-level loader while providing an extensible foundation for future
work.

The main entry point is :func:`prepare_model_table` which orchestrates the
loading, validation and transformation steps and returns both the modeling
DataFrame and minimal metadata about the ingestion.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import warnings
from typing import List, Tuple

import pandas as pd

from .spec_schema import Spec, load_and_validate_spec


def load_csvs(csv_glob: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load all CSV files matching ``csv_glob`` and concatenate them.

    Parameters
    ----------
    csv_glob:
        Glob pattern pointing to CSV files.

    Returns
    -------
    tuple
        ``(df, paths)`` where ``df`` is the concatenated DataFrame and ``paths``
        is the list of file paths that were loaded.

    Raises
    ------
    FileNotFoundError
        If no files match the provided glob.
    """
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"No CSV files found for pattern: {csv_glob}")

    dataframes = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_path"] = path
        # Derive season from ``final-<year>.csv`` filename for parity
        match = re.search(r"final-(\d{4})", os.path.basename(path))
        if match:
            df["season"] = int(match.group(1))
        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)
    return combined, paths


def validate_schema(df: pd.DataFrame, spec: Spec) -> None:
    """Validate required columns defined by the spec."""
    missing_ids = [col for col in spec.id_cols if col not in df.columns]
    if missing_ids:
        raise ValueError(f"Missing required id_cols: {missing_ids}")

    if spec.target.name not in df.columns and (
        not spec.target.derive or spec.target.derive.type != "from_scores"
    ):
        raise ValueError(
            f"Target column '{spec.target.name}' not found and no derivation provided"
        )

    # Warn for metadata columns referenced in exclude patterns that are absent
    for pattern in spec.features.exclude_regex:
        regex = re.compile(pattern)
        matched = [c for c in df.columns if regex.search(c)]
        if not matched:
            warnings.warn(
                f"No columns matched exclude pattern '{pattern}'.", stacklevel=2
            )


def align_team_perspective(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Align team perspective according to the spec.

    For Phase A the input data already follows the ``home_away`` canonical form
    so this function currently passes the data through unchanged.
    """
    if spec.team_alignment.canonical != "home_away":
        raise NotImplementedError(
            f"Unsupported canonical perspective: {spec.team_alignment.canonical}"
        )
    # Placeholder for future mapping logic when different column naming is used.
    return df


def derive_target(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Ensure the target column exists, deriving it if necessary."""
    target_col = spec.target.name
    if target_col in df.columns:
        df[target_col] = df[target_col].astype(int)
        if df[target_col].isna().any():
            raise ValueError(f"Target column '{target_col}' contains null values")
        return df

    derive = spec.target.derive
    if not derive or derive.type != "from_scores":
        raise ValueError(
            f"Target column '{target_col}' missing and no derivation available"
        )

    src = derive.source
    if src.team1_score_col not in df.columns or src.team2_score_col not in df.columns:
        raise ValueError(
            "Score columns required for target derivation are missing from data"
        )
    df[target_col] = (
        df[src.team1_score_col] > df[src.team2_score_col]
    ).astype(int)
    return df


def aggregate_to_modeling_grain(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Aggregate the data to the requested modeling grain."""
    if spec.data_grain == spec.modeling_grain:
        return df

    agg_instructions = {}
    reducers = spec.aggregation.reducers
    for reducer_name, cols in reducers.model_dump().items():
        for col in cols:
            agg_instructions[col] = reducer_name

    grouped = df.groupby(spec.aggregation.by).agg(agg_instructions).reset_index()
    return grouped


def select_features(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Apply feature inclusion/exclusion based on regex rules."""
    include_patterns = [re.compile(p) for p in spec.features.include_regex]
    exclude_patterns = [re.compile(p) for p in spec.features.exclude_regex]

    included = [
        col
        for col in df.columns
        if any(p.search(col) for p in include_patterns)
        and not any(p.search(col) for p in exclude_patterns)
    ]

    # Ensure ID columns and target are preserved
    for col in spec.id_cols:
        if col in df.columns and col not in included:
            included.append(col)
    if spec.target.name in df.columns and spec.target.name not in included:
        included.append(spec.target.name)

    feature_cols = [c for c in included if c not in spec.id_cols + [spec.target.name]]
    ordered_cols = spec.id_cols + feature_cols + [spec.target.name]
    return df[ordered_cols]


def fingerprint_inputs(paths: List[str]) -> str:
    """Compute a stable fingerprint for the input files."""
    md5 = hashlib.md5()
    for path in sorted(paths):
        stat = os.stat(path)
        md5.update(path.encode())
        md5.update(str(stat.st_size).encode())
        md5.update(str(int(stat.st_mtime)).encode())
    return md5.hexdigest()


def prepare_model_table(spec_path: str) -> Tuple[pd.DataFrame, dict]:
    """Prepare the modeling table as described by ``spec_path``.

    Parameters
    ----------
    spec_path:
        Path to the YAML spec file.

    Returns
    -------
    tuple
        ``(df, meta)`` where ``df`` is the modeling DataFrame and ``meta``
        contains dataset metadata such as feature names and an input fingerprint.
    """
    spec = load_and_validate_spec(spec_path)
    df, paths = load_csvs(spec.csv_glob)
    validate_schema(df, spec)
    df = align_team_perspective(df, spec)
    df = derive_target(df, spec)
    df = aggregate_to_modeling_grain(df, spec)
    df = select_features(df, spec)

    feature_list = [c for c in df.columns if c not in spec.id_cols + [spec.target.name]]
    meta = {
        "dataset_name": spec.dataset_name,
        "data_grain": spec.data_grain,
        "modeling_grain": spec.modeling_grain,
        "feature_list": feature_list,
        "input_fingerprint": fingerprint_inputs(paths),
    }
    return df, meta
