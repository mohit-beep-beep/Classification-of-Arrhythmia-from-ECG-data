#!/usr/bin/env python3
"""Run arrhythmia ML experiment variants and persist results to SQLite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arrhythmia_ml.config import (  # noqa: E402
    ALL_VARIANTS,
    RANDOM_STATE_MODEL,
    RANDOM_STATE_SPLIT,
    VARIANT_LABELS,
)
from arrhythmia_ml.db import (  # noqa: E402
    finish_experiment,
    init_db,
    create_experiment,
    persist_variant_outcome,
)
from arrhythmia_ml.pipeline import run_variant  # noqa: E402
from arrhythmia_ml.preprocess import preprocess  # noqa: E402


def _git_commit(repo: Path):
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train arrhythmia classifiers and write metrics to SQLite."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "Data" / "arrhythmia.csv",
        help="Path to arrhythmia CSV",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "artifacts" / "experiments.db",
        help="SQLite database path",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="arrhythmia_experiment",
        help="Experiment run name",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(ALL_VARIANTS),
        choices=list(ALL_VARIANTS),
        help="Variants to run",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=(
            "Train-only mean imputation; train-only oversampling; PCA 0.98; "
            f"split_seed={RANDOM_STATE_SPLIT}; model_seed={RANDOM_STATE_MODEL}"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset = args.dataset.resolve()
    db_path = args.db.resolve()

    print(f"Loading dataset: {dataset}")
    X, y, meta = preprocess(dataset)
    print(f"Preprocessed: {meta['n_samples']} samples, {meta['n_features']} features")
    print("Variants:")
    for v in args.variants:
        print(f"  - {v}: {VARIANT_LABELS[v]}")

    conn = init_db(db_path)
    experiment_id = create_experiment(
        conn,
        run_name=args.run_name,
        dataset_path=str(dataset),
        n_samples=meta["n_samples"],
        n_features_raw=meta["n_features_raw"],
        git_commit=_git_commit(ROOT),
        notes=args.notes,
    )
    print(f"Created experiment_id={experiment_id}")

    for variant in args.variants:
        print(f"Running variant: {VARIANT_LABELS[variant]} ...")
        outcome = run_variant(
            X,
            y,
            variant=variant,
            imputer_strategy=meta["imputer_strategy"],
            drop_column_index=meta["drop_column_index"],
            random_state_split=RANDOM_STATE_SPLIT,
            random_state_model=RANDOM_STATE_MODEL,
        )
        config_id = persist_variant_outcome(conn, experiment_id, outcome)
        best = max(
            outcome["models"],
            key=lambda m: m["metrics"]["test"]["f1_weighted"],
        )
        print(
            f"  config_id={config_id} | best test F1: "
            f"{best['model_name']}={best['metrics']['test']['f1_weighted']:.4f}"
        )

    finish_experiment(conn, experiment_id)
    conn.close()
    print(f"Done. Results written to {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
