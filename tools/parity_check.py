"""Parity check harness to compare legacy and spec data paths.

This script runs the existing ``main_mlflow.py`` twice (legacy and spec
variants) and compares counts, prevalence, feature schema and model metrics.
It exits with a non-zero status if parity checks fail.

Assumptions
-----------
* Uses the same experiment name for all runs.
* Relies on ``data/processed/<run_id>`` for split artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import mlflow

# Ensure project root's src/ is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from config.model_config import TARGET_COLUMN  # noqa: E402
from evaluation.parity_utils import (  # noqa: E402
    compare_metrics,
    compare_schema,
    compute_feature_set,
    compute_prevalence,
    load_mlflow_metrics,
    load_split_parquet,
    render_markdown_report,
    save_report_and_json,
)

METADATA_COLS = ["season", "week", "kickoff_ts", "home_team", "away_team"]

DEFAULT_TOLERANCES = {
    "metrics": {
        "accuracy": 0.01,
        "log_loss": 0.02,
        "brier_score": 0.02,
        "roc_auc": 0.01,
        "ece_10": 0.02,
        "ece_15": 0.02,
    },
    "counts": 0,
    "prevalence": 0.005,
}


def parse_tolerances(arg: str | None) -> Dict[str, Dict[str, float]]:
    if not arg:
        return DEFAULT_TOLERANCES
    try:
        custom = json.loads(arg)
    except json.JSONDecodeError:
        custom = {}
        for part in arg.split(','):
            if '=' not in part:
                continue
            k, v = part.split('=', 1)
            custom[k.strip()] = float(v)
    tol = json.loads(json.dumps(DEFAULT_TOLERANCES))  # deep copy
    for k, v in custom.items():
        if k in ('counts', 'prevalence'):
            tol[k] = float(v)
        else:
            tol['metrics'][k] = float(v)
    return tol


def run_pipeline(exp_id: str, args, data_spec: str | None = None) -> str:
    cmd = [
        sys.executable,
        'src/main_mlflow.py',
        '--experiment-name', args.experiment_name,
        '--train-years', args.train_years,
        '--eval-years', args.eval_years,
        '--test-years', args.test_years,
        '--models', args.models,
        '--run-type', 'parity',
    ]
    data_source = 'legacy'
    if data_spec:
        cmd.extend(['--data-spec', data_spec])
        data_source = 'spec'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run(cmd, check=True, env=env)
    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.run_type = 'parity' and tags.data_source = '{data_source}'",
        order_by=['start_time DESC'],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No MLflow run found for data_source={data_source}")
    return runs.loc[0, 'run_id']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check parity between legacy and spec data paths")
    p.add_argument('--train-years', required=True, help='Comma-separated training years')
    p.add_argument('--eval-years', required=True, help='Comma-separated eval years')
    p.add_argument('--test-years', required=True, help='Comma-separated test years')
    p.add_argument('--models', default='rf,svm,mlp,xgb', help='Comma-separated models')
    p.add_argument('--spec', required=True, help='Path to data spec YAML')
    p.add_argument('--experiment-name', default='Parity_Check')
    p.add_argument('--tolerances', default=None, help='JSON or comma-separated tolerances')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tolerances = parse_tolerances(args.tolerances)
    exp = mlflow.get_experiment_by_name(args.experiment_name)
    if exp is None:
        mlflow.set_experiment(args.experiment_name)
        exp = mlflow.get_experiment_by_name(args.experiment_name)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    with mlflow.start_run(run_name=f"Parity_{timestamp}") as pr:
        legacy_run_id = run_pipeline(exp.experiment_id, args, data_spec=None)
        spec_run_id = run_pipeline(exp.experiment_id, args, data_spec=args.spec)
        mlflow.log_params(
            {
                'legacy_run_id': legacy_run_id,
                'spec_run_id': spec_run_id,
                'parity_tolerances': json.dumps(tolerances),
            }
        )
        mlflow.set_tags({'data_source_legacy': 'legacy', 'data_source_spec': 'spec'})

        metrics_legacy = load_mlflow_metrics(legacy_run_id)
        metrics_spec = load_mlflow_metrics(spec_run_id)
        splits_legacy = load_split_parquet(legacy_run_id)
        splits_spec = load_split_parquet(spec_run_id)

        counts = {}
        prevalence = {}
        for split in ('train', 'eval', 'test'):
            l_df = splits_legacy[split]
            s_df = splits_spec[split]
            counts[split] = {
                'legacy': len(l_df),
                'spec': len(s_df),
                'delta': len(s_df) - len(l_df),
            }
            prevalence[split] = {
                'legacy': compute_prevalence(l_df, TARGET_COLUMN),
                'spec': compute_prevalence(s_df, TARGET_COLUMN),
            }
            prevalence[split]['delta'] = prevalence[split]['spec'] - prevalence[split]['legacy']

        features_legacy, hash_legacy = compute_feature_set(
            splits_legacy['train'], TARGET_COLUMN, METADATA_COLS
        )
        features_spec, hash_spec = compute_feature_set(
            splits_spec['train'], TARGET_COLUMN, METADATA_COLS
        )
        # ``feature_info`` is meant for serialization, so ensure all values are JSON
        # friendly (i.e., convert sets to sorted lists).
        feature_info = {
            'legacy': sorted(features_legacy),
            'spec': sorted(features_spec),
            'hash_legacy': hash_legacy,
            'hash_spec': hash_spec,
            'count_legacy': len(features_legacy),
            'count_spec': len(features_spec),
            'diff_legacy_only': sorted(features_legacy - features_spec),
            'diff_spec_only': sorted(features_spec - features_legacy),
        }

        counts_pair = {k: (v['legacy'], v['spec']) for k, v in counts.items()}
        prevalence_pair = {k: (v['legacy'], v['spec']) for k, v in prevalence.items()}
        features_pair = {
            'legacy': features_legacy,
            'spec': features_spec,
            'hash_legacy': hash_legacy,
            'hash_spec': hash_spec,
        }

        schema_failures = compare_schema(
            counts_pair, prevalence_pair, features_pair, tolerances['counts'], tolerances['prevalence']
        )
        metric_failures = compare_metrics(
            metrics_legacy, metrics_spec, tolerances['metrics']
        )

        metrics_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for model in metrics_legacy:
            metrics_summary[model] = {}
            for metric, tol in tolerances['metrics'].items():
                l_val = metrics_legacy.get(model, {}).get(metric)
                s_val = metrics_spec.get(model, {}).get(metric)
                if l_val is None or s_val is None:
                    continue
                # Cast to plain ``float`` to ensure JSON serialization later.
                l_val = float(l_val)
                s_val = float(s_val)
                delta = s_val - l_val
                metrics_summary[model][metric] = {
                    'legacy': l_val,
                    'spec': s_val,
                    'delta': delta,
                    'tolerance': float(tol),
                    'passed': abs(delta) <= float(tol),
                }

        all_failures = schema_failures + metric_failures
        conclusion = 'PARITY PASSED' if not all_failures else 'PARITY FAILED: ' + '; '.join(all_failures)

        summary = {
            'legacy_run_id': legacy_run_id,
            'spec_run_id': spec_run_id,
            'counts': counts,
            'prevalence': prevalence,
            'features': feature_info,
            'metrics': metrics_summary,
            'conclusion': conclusion,
        }

        report_md = render_markdown_report(
            legacy_run_id,
            spec_run_id,
            counts,
            prevalence,
            feature_info,
            metrics_summary,
            tolerances,
            conclusion,
        )
        save_report_and_json(pr.info.run_id, report_md, summary)

        if all_failures:
            for msg in all_failures:
                print(msg)
            raise SystemExit(1)
        else:
            print('Parity check passed.')


if __name__ == '__main__':
    main()