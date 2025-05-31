import os
import pandas as pd
from glob import glob

def load_nfl_data(data_dir='nfl_data'):
    """
    Loads all 'final-<year>.csv' files from subdirectories in the given data_dir.
    Returns a single combined DataFrame if all files have matching columns.
    """
    combined_df = pd.DataFrame()
    file_paths = glob(os.path.join(data_dir, '*', 'final-*.csv'))

    dataframes = []
    columns_set = None

    for file_path in sorted(file_paths):
        try:
            df = pd.read_csv(file_path)
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