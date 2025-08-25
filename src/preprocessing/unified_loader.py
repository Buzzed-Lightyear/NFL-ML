"""Unified dataset loader using schema-driven specs.

This module exposes a thin wrapper around :func:`src.data.adapter.prepare_model_table`
so that other parts of the project can load datasets without being coupled to
the underlying adapter implementation.

Example
-------
>>> from src.preprocessing.unified_loader import load_dataset_from_spec
>>> df, meta = load_dataset_from_spec('src/data/specs/game_parity.yaml')
>>> df.shape
(284, 56)
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from data.adapter import prepare_model_table


def load_dataset_from_spec(spec_path: str) -> Tuple[pd.DataFrame, dict]:
    """Load a dataset described by ``spec_path``.

    Parameters
    ----------
    spec_path: str
        Path to the YAML spec file.

    Returns
    -------
    tuple
        ``(df, meta)`` where ``df`` is the modeling DataFrame and ``meta``
        contains metadata such as the feature list and an input fingerprint.
    """
    return prepare_model_table(spec_path)