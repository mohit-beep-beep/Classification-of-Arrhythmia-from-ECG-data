"""Tests for SQL analytics helpers."""

from __future__ import annotations

from arrhythmia_ml import analytics
from arrhythmia_ml.db import create_experiment, finish_experiment, persist_variant_outcome
from arrhythmia_ml.pipeline import run_all_variants


def _seed(conn, synthetic_xy):
    X, y = synthetic_xy
    exp_id = create_experiment(
        conn,
        run_name="analytics_seed",
        dataset_path="synth",
        n_samples=len(y),
        n_features_raw=X.shape[1],
    )
    for outcome in run_all_variants(X, y, drop_column_index=None):
        persist_variant_outcome(conn, exp_id, outcome)
    finish_experiment(conn, exp_id)
    return exp_id


def test_analytics_queries(temp_db, synthetic_xy):
    conn, _ = temp_db
    exp_id = _seed(conn, synthetic_xy)

    board = analytics.leaderboard(conn, experiment_id=exp_id)
    assert set(board.columns) >= {"variant", "model_name", "score"}
    assert len(board) == 4 * 6  # 4 variants × 6 models

    best = analytics.best_model_per_variant(conn, experiment_id=exp_id)
    assert len(best) == 4
    assert set(best["variant"]) == {"baseline", "pca", "ros_pca", "smote_pca"}

    svc = analytics.compare_model_across_variants(
        conn, model_name="Kernelized SVC", experiment_id=exp_id
    )
    assert len(svc) == 4

    gaps = analytics.train_test_accuracy_gap(conn, experiment_id=exp_id)
    assert "accuracy_gap" in gaps.columns
    assert len(gaps) == 24

    pca_df = analytics.pca_components_by_variant(conn, experiment_id=exp_id)
    assert len(pca_df) == 3  # pca, ros_pca, smote_pca

    bal = analytics.class_balance_impact(conn, experiment_id=exp_id)
    assert set(bal["stage"]).issubset({"train_before", "train_after"})

    hot = analytics.confusion_hotspots(conn, experiment_id=exp_id, limit=5)
    assert len(hot) <= 5

    summary = analytics.latest_experiment_summary(conn)
    assert len(summary) == 24
    assert summary["experiment_id"].nunique() == 1
