import os
import pandas as pd
from glob import glob
import re
from typing import Iterable, Optional


def load_nfl_data(data_dir: str = 'data/processed', years: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Load all ``final-<year>.csv`` files for the requested years.

    Parameters
    ----------
    data_dir: str
        Base directory containing year subdirectories with ``final-<year>.csv`` files.
    years: Iterable[int], optional
        If provided, only files whose season matches one of these years will be
        loaded.

    Returns
    -------
    pd.DataFrame
        Combined data for the selected seasons with a ``season`` column added.
    """
    combined_df = pd.DataFrame()
    file_paths = glob(os.path.join(data_dir, '*', 'final-*.csv'))

    dataframes = []
    columns_set = None

    for file_path in sorted(file_paths):
        match = re.search(r'final-(\d{4})', os.path.basename(file_path))
        year = int(match.group(1)) if match else None
        if years is not None and year not in years:
            continue

        try:
            df = pd.read_csv(file_path)
            df['season'] = year
            print(f"Loaded {file_path}, shape: {df.shape}")

            if columns_set is None:
                columns_set = list(df.columns)
            elif list(df.columns) != columns_set:
                print(f"ERROR: Column mismatch in {file_path}, skipping.")
                continue

            dataframes.append(df)

        except FileNotFoundError:
            print(f"ERROR: File not found: {file_path}")
        except Exception as e:
            print(f"ERROR: Could not load {file_path}: {e}")

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\nCombined DataFrame shape: {combined_df.shape}")
    else:
        print("\nNo valid data files loaded.")

    return combined_df


def load_via_spec(spec_path: str) -> pd.DataFrame:
    """Load data using a YAML spec via the unified adapter.

    This helper preserves backwards compatibility while providing an entry
    point for the new schema-driven loader. The adapter returns both the
    modeling DataFrame and metadata, but this wrapper mirrors the old loader
    by returning only the DataFrame.

    Parameters
    ----------
    spec_path: str
        Path to the dataset specification file.

    Returns
    -------
    pd.DataFrame
        Modeling table as produced by the adapter.
    """
    from .unified_loader import load_dataset_from_spec

    df, _ = load_dataset_from_spec(spec_path)
    return df