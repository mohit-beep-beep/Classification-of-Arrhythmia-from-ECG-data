"""Tests for SQLite persistence."""

from __future__ import annotations

import sqlite3

import pytest

from arrhythmia_ml.db import (
    create_experiment,
    finish_experiment,
    insert_config,
    insert_metrics,
    insert_model_run,
    persist_variant_outcome,
)
from arrhythmia_ml.pipeline import run_variant


def test_schema_creates_tables(temp_db):
    conn, _ = temp_db
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    expected = {
        "experiments",
        "experiment_configs",
        "model_runs",
        "metrics",
        "confusion_matrices",
        "class_distributions",
    }
    assert expected.issubset(tables)


def test_insert_experiment_metrics_roundtrip(temp_db):
    conn, _ = temp_db
    exp_id = create_experiment(
        conn,
        run_name="unit_test",
        dataset_path="fixture.csv",
        n_samples=10,
        n_features_raw=5,
    )
    config_id = insert_config(
        conn,
        exp_id,
        {
            "variant": "baseline",
            "test_size": 0.2,
            "random_state_split": 1,
            "imputer_strategy": "mean",
            "drop_column_index": 13,
            "scaler": "StandardScaler",
            "pca_variance": None,
            "pca_n_components": None,
            "oversampler": None,
            "oversample_scope": None,
            "train_rows_before": 8,
            "train_rows_after": 8,
            "test_rows": 2,
        },
    )
    run_id = insert_model_run(
        conn,
        config_id,
        model_name="KNN Classifier",
        sklearn_class="sklearn.neighbors.KNeighborsClassifier",
        hyperparameters={"n_neighbors": 5},
    )
    insert_metrics(
        conn,
        run_id,
        {
            "train": {"accuracy": 0.9, "f1_weighted": 0.88},
            "test": {"accuracy": 0.7, "f1_weighted": 0.65},
        },
    )
    finish_experiment(conn, exp_id)

    row = conn.execute(
        """
        SELECT metric_value FROM metrics
        WHERE run_id = ? AND split = 'test' AND metric_name = 'accuracy'
        """,
        (run_id,),
    ).fetchone()
    assert row[0] == pytest.approx(0.7)

    finished = conn.execute(
        "SELECT finished_at FROM experiments WHERE experiment_id = ?",
        (exp_id,),
    ).fetchone()[0]
    assert finished is not None


def test_metrics_unique_constraint(temp_db):
    conn, _ = temp_db
    exp_id = create_experiment(conn, run_name="dup", dataset_path="x.csv")
    config_id = insert_config(
        conn,
        exp_id,
        {
            "variant": "baseline",
            "test_size": 0.2,
            "random_state_split": 1,
            "imputer_strategy": "mean",
        },
    )
    run_id = insert_model_run(
        conn,
        config_id,
        model_name="Decision Tree",
        sklearn_class="sklearn.tree.DecisionTreeClassifier",
        hyperparameters={},
    )
    insert_metrics(conn, run_id, {"test": {"accuracy": 0.5}})
    with pytest.raises(sqlite3.IntegrityError):
        insert_metrics(conn, run_id, {"test": {"accuracy": 0.6}})


def test_persist_variant_outcome(temp_db, synthetic_xy):
    conn, _ = temp_db
    X, y = synthetic_xy
    outcome = run_variant(X, y, variant="baseline", drop_column_index=None)
    exp_id = create_experiment(conn, run_name="persist", dataset_path="synth")
    config_id = persist_variant_outcome(conn, exp_id, outcome)
    n_models = conn.execute(
        "SELECT COUNT(*) FROM model_runs WHERE config_id = ?", (config_id,)
    ).fetchone()[0]
    assert n_models == 6
    n_metrics = conn.execute(
        """
        SELECT COUNT(*) FROM metrics met
        JOIN model_runs m ON m.run_id = met.run_id
        WHERE m.config_id = ?
        """,
        (config_id,),
    ).fetchone()[0]
    # 6 models * 2 splits * 4 metrics
    assert n_metrics == 48
