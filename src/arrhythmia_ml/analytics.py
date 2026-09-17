"""Named SQL analytics helpers over the experiments SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .db import connect

PathLike = Union[str, Path]


def _read_sql(conn: sqlite3.Connection, sql: str, params=None) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params or [])


def leaderboard(
    conn: sqlite3.Connection,
    *,
    metric_name: str = "f1_weighted",
    split: str = "test",
    experiment_id: Optional[int] = None,
) -> pd.DataFrame:
    sql = """
    SELECT e.experiment_id, e.run_name, c.variant, c.variant_label, m.model_name,
           met.metric_value AS score
    FROM metrics met
    JOIN model_runs m ON m.run_id = met.run_id
    JOIN experiment_configs c ON c.config_id = m.config_id
    JOIN experiments e ON e.experiment_id = c.experiment_id
    WHERE met.split = ? AND met.metric_name = ?
    """
    params = [split, metric_name]
    if experiment_id is not None:
        sql += " AND e.experiment_id = ?"
        params.append(experiment_id)
    sql += " ORDER BY met.metric_value DESC"
    return _read_sql(conn, sql, params)


def best_model_per_variant(
    conn: sqlite3.Connection,
    *,
    metric_name: str = "f1_weighted",
    split: str = "test",
    experiment_id: Optional[int] = None,
) -> pd.DataFrame:
    sql = """
    WITH ranked AS (
      SELECT e.experiment_id, e.run_name, c.variant, c.variant_label, m.model_name,
             met.metric_value AS score,
             ROW_NUMBER() OVER (
               PARTITION BY e.experiment_id, c.variant
               ORDER BY met.metric_value DESC
             ) AS rn
      FROM metrics met
      JOIN model_runs m ON m.run_id = met.run_id
      JOIN experiment_configs c ON c.config_id = m.config_id
      JOIN experiments e ON e.experiment_id = c.experiment_id
      WHERE met.split = ? AND met.metric_name = ?
    """
    params = [split, metric_name]
    if experiment_id is not None:
        sql += " AND e.experiment_id = ?"
        params.append(experiment_id)
    sql += """
    )
    SELECT experiment_id, run_name, variant, variant_label, model_name, score
    FROM ranked
    WHERE rn = 1
    ORDER BY variant
    """
    return _read_sql(conn, sql, params)


def compare_model_across_variants(
    conn: sqlite3.Connection,
    *,
    model_name: str = "Kernelized SVC",
    metric_name: str = "f1_weighted",
    split: str = "test",
    experiment_id: Optional[int] = None,
) -> pd.DataFrame:
    sql = """
    SELECT e.experiment_id, e.run_name, c.variant, c.variant_label, m.model_name,
           met.metric_value AS score
    FROM metrics met
    JOIN model_runs m ON m.run_id = met.run_id
    JOIN experiment_configs c ON c.config_id = m.config_id
    JOIN experiments e ON e.experiment_id = c.experiment_id
    WHERE m.model_name = ? AND met.split = ? AND met.metric_name = ?
    """
    params = [model_name, split, metric_name]
    if experiment_id is not None:
        sql += " AND e.experiment_id = ?"
        params.append(experiment_id)
    sql += " ORDER BY c.variant"
    return _read_sql(conn, sql, params)


def train_test_accuracy_gap(
    conn: sqlite3.Connection,
    *,
    experiment_id: Optional[int] = None,
) -> pd.DataFrame:
    sql = """
    SELECT e.experiment_id, c.variant, c.variant_label, m.model_name,
           train.metric_value AS train_accuracy,
           test.metric_value AS test_accuracy,
           (train.metric_value - test.metric_value) AS accuracy_gap
    FROM model_runs m
    JOIN experiment_configs c ON c.config_id = m.config_id
    JOIN experiments e ON e.experiment_id = c.experiment_id
    JOIN metrics train
      ON train.run_id = m.run_id
     AND train.split = 'train'
     AND train.metric_name = 'accuracy'
    JOIN metrics test
      ON test.run_id = m.run_id
     AND test.split = 'test'
     AND test.metric_name = 'accuracy'
    """
    params = []
    if experiment_id is not None:
        sql += " WHERE e.experiment_id = ?"
        params.append(experiment_id)
    sql += " ORDER BY accuracy_gap DESC"
    return _read_sql(conn, sql, params)


def pca_components_by_variant(
    conn: sqlite3.Connection,
    *,
    experiment_id: Optional[int] = None,
) -> pd.DataFrame:
    sql = """
    SELECT e.experiment_id, c.variant, c.variant_label, c.pca_variance,
           c.pca_n_components, c.train_rows_before, c.train_rows_after, c.test_rows,
           c.random_state_split, c.random_state_model
    FROM experiment_configs c
    JOIN experiments e ON e.experiment_id = c.experiment_id
    WHERE c.pca_n_components IS NOT NULL
    """
    params = []
    if experiment_id is not None:
        sql += " AND e.experiment_id = ?"
        params.append(experiment_id)
    sql += " ORDER BY c.variant"
    return _read_sql(conn, sql, params)


def class_balance_impact(
    conn: sqlite3.Connection,
    *,
    config_id: Optional[int] = None,
    experiment_id: Optional[int] = None,
) -> pd.DataFrame:
    sql = """
    SELECT c.config_id, c.variant, c.variant_label, d.stage, d.class_label, d.count
    FROM class_distributions d
    JOIN experiment_configs c ON c.config_id = d.config_id
    JOIN experiments e ON e.experiment_id = c.experiment_id
    WHERE d.stage IN ('train_before', 'train_after')
    """
    params = []
    if config_id is not None:
        sql += " AND c.config_id = ?"
        params.append(config_id)
    if experiment_id is not None:
        sql += " AND e.experiment_id = ?"
        params.append(experiment_id)
    sql += " ORDER BY c.variant, d.stage, d.class_label"
    return _read_sql(conn, sql, params)


def confusion_hotspots(
    conn: sqlite3.Connection,
    *,
    run_id: Optional[int] = None,
    experiment_id: Optional[int] = None,
    limit: int = 20,
) -> pd.DataFrame:
    sql = """
    SELECT e.experiment_id, c.variant, c.variant_label, m.model_name, m.run_id,
           cm.true_label, cm.pred_label, cm.count
    FROM confusion_matrices cm
    JOIN model_runs m ON m.run_id = cm.run_id
    JOIN experiment_configs c ON c.config_id = m.config_id
    JOIN experiments e ON e.experiment_id = c.experiment_id
    WHERE cm.true_label != cm.pred_label
    """
    params = []
    if run_id is not None:
        sql += " AND m.run_id = ?"
        params.append(run_id)
    if experiment_id is not None:
        sql += " AND e.experiment_id = ?"
        params.append(experiment_id)
    sql += " ORDER BY cm.count DESC LIMIT ?"
    params.append(limit)
    return _read_sql(conn, sql, params)


def latest_experiment_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = """
    SELECT e.experiment_id, e.run_name, e.started_at, e.finished_at,
           c.variant, c.variant_label, m.model_name,
           met.metric_name, met.metric_value AS test_score
    FROM experiments e
    JOIN experiment_configs c ON c.experiment_id = e.experiment_id
    JOIN model_runs m ON m.config_id = c.config_id
    JOIN metrics met ON met.run_id = m.run_id
    WHERE e.experiment_id = (SELECT MAX(experiment_id) FROM experiments)
      AND met.split = 'test'
      AND met.metric_name = 'f1_weighted'
    ORDER BY c.variant, met.metric_value DESC
    """
    return _read_sql(conn, sql)


def load_db(db_path: PathLike, *, read_only: bool = True) -> sqlite3.Connection:
    """Open the experiments DB. Defaults to read-only (for Streamlit / analytics)."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found: {path}. Run scripts/run_experiments.py first."
        )
    return connect(path, read_only=read_only)
