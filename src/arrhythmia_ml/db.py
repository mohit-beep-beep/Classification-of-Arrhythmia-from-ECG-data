"""SQLite persistence for ML experiment results."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

PathLike = Union[str, Path]

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: PathLike, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if read_only:
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {path}")
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: PathLike, schema_path: Optional[PathLike] = None) -> sqlite3.Connection:
    """Create tables if they do not exist and return a connection."""
    schema_file = Path(schema_path) if schema_path else SCHEMA_PATH
    ddl = schema_file.read_text(encoding="utf-8")
    conn = connect(db_path, read_only=False)
    conn.executescript(ddl)
    _ensure_config_columns(conn)
    conn.commit()
    return conn


def _ensure_config_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first schema revision (idempotent)."""
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(experiment_configs)").fetchall()
    }
    if "random_state_model" not in cols:
        conn.execute(
            "ALTER TABLE experiment_configs ADD COLUMN random_state_model INTEGER"
        )
    if "variant_label" not in cols:
        conn.execute("ALTER TABLE experiment_configs ADD COLUMN variant_label TEXT")


def create_experiment(
    conn: sqlite3.Connection,
    *,
    run_name: str,
    dataset_path: str,
    n_samples: Optional[int] = None,
    n_features_raw: Optional[int] = None,
    git_commit: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO experiments (
            run_name, started_at, dataset_path, n_samples,
            n_features_raw, git_commit, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_name,
            _utc_now(),
            dataset_path,
            n_samples,
            n_features_raw,
            git_commit,
            notes,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_experiment(conn: sqlite3.Connection, experiment_id: int) -> None:
    conn.execute(
        "UPDATE experiments SET finished_at = ? WHERE experiment_id = ?",
        (_utc_now(), experiment_id),
    )
    conn.commit()


def insert_config(
    conn: sqlite3.Connection,
    experiment_id: int,
    config: Mapping[str, Any],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO experiment_configs (
            experiment_id, variant, test_size, random_state_split,
            random_state_model, variant_label, imputer_strategy,
            drop_column_index, scaler, pca_variance,
            pca_n_components, oversampler, oversample_scope,
            train_rows_before, train_rows_after, test_rows
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            experiment_id,
            config["variant"],
            config["test_size"],
            config["random_state_split"],
            config.get("random_state_model"),
            config.get("variant_label"),
            config["imputer_strategy"],
            config.get("drop_column_index"),
            config.get("scaler"),
            config.get("pca_variance"),
            config.get("pca_n_components"),
            config.get("oversampler"),
            config.get("oversample_scope"),
            config.get("train_rows_before"),
            config.get("train_rows_after"),
            config.get("test_rows"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_class_distributions(
    conn: sqlite3.Connection,
    config_id: int,
    distributions: Mapping[str, Mapping[int, int]],
) -> None:
    rows = []
    for stage, counts in distributions.items():
        for class_label, count in counts.items():
            rows.append((config_id, stage, int(class_label), int(count)))
    if rows:
        conn.executemany(
            """
            INSERT INTO class_distributions (config_id, stage, class_label, count)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def insert_model_run(
    conn: sqlite3.Connection,
    config_id: int,
    *,
    model_name: str,
    sklearn_class: str,
    hyperparameters: Mapping[str, Any],
    trained_at: Optional[str] = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO model_runs (
            config_id, model_name, sklearn_class, hyperparameters_json, trained_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            config_id,
            model_name,
            sklearn_class,
            json.dumps(hyperparameters, sort_keys=True, default=str),
            trained_at or _utc_now(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_metrics(
    conn: sqlite3.Connection,
    run_id: int,
    metrics: Mapping[str, Mapping[str, float]],
) -> None:
    """
    Persist metrics.

    Expected shape:
        {"train": {"accuracy": 0.9, ...}, "test": {"accuracy": 0.8, ...}}
    """
    rows = []
    for split, split_metrics in metrics.items():
        for metric_name, metric_value in split_metrics.items():
            rows.append((run_id, split, metric_name, float(metric_value)))
    conn.executemany(
        """
        INSERT INTO metrics (run_id, split, metric_name, metric_value)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def insert_confusion_matrix(
    conn: sqlite3.Connection,
    run_id: int,
    cm_rows: Sequence[tuple[int, int, int]],
) -> None:
    """Insert flattened confusion matrix rows: (true_label, pred_label, count)."""
    if not cm_rows:
        return
    conn.executemany(
        """
        INSERT INTO confusion_matrices (run_id, true_label, pred_label, count)
        VALUES (?, ?, ?, ?)
        """,
        [(run_id, int(t), int(p), int(c)) for t, p, c in cm_rows],
    )
    conn.commit()


def persist_variant_outcome(
    conn: sqlite3.Connection,
    experiment_id: int,
    outcome: Mapping[str, Any],
) -> int:
    """
    Persist a full variant outcome from pipeline.run_variant.

    Returns config_id.
    """
    config_id = insert_config(conn, experiment_id, outcome["config"])
    insert_class_distributions(conn, config_id, outcome.get("distributions", {}))
    for model_result in outcome.get("models", []):
        run_id = insert_model_run(
            conn,
            config_id,
            model_name=model_result["model_name"],
            sklearn_class=model_result["sklearn_class"],
            hyperparameters=model_result["hyperparameters"],
            trained_at=model_result.get("trained_at"),
        )
        insert_metrics(conn, run_id, model_result["metrics"])
        insert_confusion_matrix(conn, run_id, model_result.get("cm_rows", []))
    return config_id
