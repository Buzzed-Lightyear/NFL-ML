"""Compare recent MLflow runs by metrics."""
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import mlflow

METRIC_COLUMNS = [
    'roc_auc',
    'log_loss',
    'brier_score',
    'accuracy',
    'calibration_error',
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare recent MLflow runs")
    parser.add_argument('--experiment-name', default='NFL_Game_Outcome_Prediction')
    parser.add_argument('--split-id', required=True)
    parser.add_argument('--feature-hash', required=True)
    parser.add_argument('-n', '--num-runs', type=int, default=5)
    parser.add_argument('--baseline-run-id', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    exp = mlflow.get_experiment_by_name(args.experiment_name)
    if exp is None:
        raise ValueError(f"Experiment {args.experiment_name} not found")

    filter_str = f"tags.split_id = '{args.split_id}' and tags.feature_hash = '{args.feature_hash}'"
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                              filter_string=filter_str,
                              order_by=['start_time DESC'],
                              max_results=args.num_runs)

    if runs.empty:
        print("No runs found matching criteria")
        return

    runs = runs[runs['tags.model'].notna()]

    records = []
    for _, row in runs.iterrows():
        record = {
            'run_id': row['run_id'],
            'model': row['tags.model'],
        }
        for m in METRIC_COLUMNS:
            record[m] = row.get(f'metrics.{m}')
        record['run_type'] = row.get('tags.run_type')
        records.append(record)

    df = pd.DataFrame(records)
    baseline_df = df[df['run_type'] == 'baseline'].groupby('model').first()[METRIC_COLUMNS]

    comparison_rows = []
    for run_id, group in df.groupby('run_id'):
        metrics = group.set_index('model')[METRIC_COLUMNS]
        deltas = metrics - baseline_df
        deltas = deltas.add_prefix('delta_')
        combined = pd.concat([metrics, deltas], axis=1)
        combined['run_id'] = run_id
        comparison_rows.append(combined.reset_index())

    comparison = pd.concat(comparison_rows, ignore_index=True)
    print(comparison)

    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"run_comparison_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    comparison.to_csv(out_path, index=False)
    print(f"Saved comparison to {out_path}")


if __name__ == '__main__':
    main()
