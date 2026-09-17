"""Ensure DB metrics match sklearn recomputation — no fabricated values."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score

from arrhythmia_ml.db import create_experiment, persist_variant_outcome
from arrhythmia_ml.pipeline import run_variant


def test_db_metrics_match_sklearn_predictions(temp_db, synthetic_xy):
    conn, _ = temp_db
    X, y = synthetic_xy
    outcome = run_variant(X, y, variant="pca", drop_column_index=None)

    # Recompute from stored predictions before/with persistence
    for model in outcome["models"]:
        y_true = model["y_test"]
        y_pred = model["y_test_pred"]
        expected_acc = accuracy_score(y_true, y_pred)
        expected_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        assert model["metrics"]["test"]["accuracy"] == expected_acc
        assert model["metrics"]["test"]["f1_weighted"] == expected_f1

    exp_id = create_experiment(conn, run_name="no_fabricate", dataset_path="synth")
    config_id = persist_variant_outcome(conn, exp_id, outcome)

    rows = conn.execute(
        """
        SELECT m.model_name, met.metric_name, met.metric_value
        FROM metrics met
        JOIN model_runs m ON m.run_id = met.run_id
        WHERE m.config_id = ? AND met.split = 'test'
          AND met.metric_name IN ('accuracy', 'f1_weighted')
        """,
        (config_id,),
    ).fetchall()

    by_key = {(r[0], r[1]): r[2] for r in rows}
    for model in outcome["models"]:
        name = model["model_name"]
        assert by_key[(name, "accuracy")] == model["metrics"]["test"]["accuracy"]
        assert by_key[(name, "f1_weighted")] == model["metrics"]["test"]["f1_weighted"]
