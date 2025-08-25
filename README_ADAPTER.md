# Adapter Layer Overview

This repository includes a schema-driven data adapter that can ingest raw CSV
files using a YAML **specification**.  The adapter enables future flexibility in
data sources while maintaining Phase‑A parity with the existing game-level
loader.

## Spec basics
A spec defines where the CSV files live, required identifier columns, how to
derive the target label, and which features to expose.  Specs are validated via
`pydantic` models in `src/data/spec_schema.py`.

Key sections:

- `csv_glob`: Glob pattern to locate CSV files.
- `id_cols`: Columns preserved as identifiers (not used as features).
- `target`: Name and derivation instructions for the label.
- `features`: Regex-based include/exclude rules.

## Parity spec
The file `src/data/specs/game_parity.yaml` reproduces the current
`load_nfl_data` output.  It loads `data/processed/*/final-*.csv` files, ensures
the `team1_win` target exists, and returns the same feature columns as the
original pipeline.

## Drive aggregation

Drive-level CSVs must include at minimum the following columns:
`game_id`, `season`, `week`, `home_team`, `away_team`, `drive_id`, `posteam` and
`defteam`. Final score columns (`home_score_final`, `away_score_final`) are used to
derive the binary target `team1_win`.

Each drive record is aligned to a home/away perspective before aggregation. If
`posteam` equals `home_team` the drive's statistics are mapped to `team1_*`
columns; otherwise they are mapped to `team2_*`. This produces a symmetric
home-versus-away table prior to aggregation.

The default reducers applied when aggregating to one row per game are defined in
`src/data/specs/drive_agg_v1.yaml`. Columns can be moved between `mean`, `sum`,
`max` and `last` reducer lists to customize aggregation without code changes.


## Usage
Run the loader with:

```bash
python -c "from src.preprocessing.unified_loader import load_dataset_from_spec; import pprint; df, meta = load_dataset_from_spec('src/data/specs/game_parity.yaml'); print(df.shape); pprint.pp(meta)"
```

`meta` includes:

- `dataset_name`
- `data_grain` and `modeling_grain`
- `feature_list`
- `input_fingerprint` – hash of input file names, sizes and timestamps

## Notes
The parity spec omits game identifiers since the provided CSVs do not contain
them. Future specs can add richer metadata or alternate grains without modifying
model code.
