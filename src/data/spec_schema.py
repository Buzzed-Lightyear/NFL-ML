"""Schema and validation utilities for dataset specification files.

This module defines pydantic models that mirror the YAML spec format used by
``src/data/adapter.py``.  The spec describes how raw CSV files should be
interpreted and transformed into a modeling table.

Usage
-----
>>> from src.data.spec_schema import load_and_validate_spec
>>> spec = load_and_validate_spec("src/data/specs/game_parity.yaml")
>>> spec.dataset_name
'game_parity'
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class ScoreSource(BaseModel):
    """Column names containing the final scores for each team."""

    team1_score_col: str
    team2_score_col: str


class TargetDerive(BaseModel):
    """Instructions on how to derive the modeling target."""

    type: Literal["existing", "from_scores"]
    source: Optional[ScoreSource] = None

    @model_validator(mode="after")
    def require_source_for_scores(cls, values: Dict) -> Dict:
        """Ensure score columns are provided when deriving from scores."""
        if values.type == "from_scores" and not values.source:
            raise ValueError("'from_scores' derivation requires a 'source' section")
        return values


class Target(BaseModel):
    """Target column description."""

    name: str
    derive: Optional[TargetDerive] = None


class TeamAlignment(BaseModel):
    """Specification for aligning team perspective to a canonical format."""

    canonical: Literal["home_away"]
    mapping: Dict[str, str] = Field(default_factory=dict)


class AggregationReducers(BaseModel):
    """Reducers to apply when aggregating to the modeling grain."""

    mean: List[str] = Field(default_factory=list)
    sum: List[str] = Field(default_factory=list)
    max: List[str] = Field(default_factory=list)
    last: List[str] = Field(default_factory=list)


class Aggregation(BaseModel):
    """Aggregation instructions for converting raw data to the modeling grain."""

    by: List[str] = Field(default_factory=list)
    reducers: AggregationReducers = Field(default_factory=AggregationReducers)


class Features(BaseModel):
    """Feature selection rules based on regular expressions."""

    include_regex: List[str] = Field(default_factory=lambda: [".*"])
    exclude_regex: List[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """Provenance configuration for the dataset."""

    allow_post_final_columns: bool = True


class Spec(BaseModel):
    """Top-level dataset specification."""

    dataset_name: str
    csv_glob: str
    data_grain: Literal["game", "drive"]
    modeling_grain: Literal["game", "drive"]
    id_cols: List[str] = Field(default_factory=list)
    target: Target
    team_alignment: TeamAlignment
    aggregation: Aggregation
    features: Features
    provenance: Provenance


def load_and_validate_spec(path: str | Path) -> Spec:
    """Load a YAML spec file and return a validated :class:`Spec` object.

    Parameters
    ----------
    path:
        Path to the YAML specification file.

    Returns
    -------
    Spec
        Parsed and validated specification instance.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValidationError
        If the YAML contents do not conform to the spec schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")

    with path.open("r") as fh:
        raw = yaml.safe_load(fh)

    try:
        return Spec(**raw)
    except ValidationError as e:
        raise ValidationError(f"Invalid spec file {path}: {e}")
