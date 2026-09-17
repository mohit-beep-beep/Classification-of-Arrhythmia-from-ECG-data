"""Shared pytest fixtures for arrhythmia_ml tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def schema_path(repo_root: Path) -> Path:
    return repo_root / "sql" / "schema.sql"


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """
    Tiny arrhythmia-like CSV: 30 rows, 16 columns (15 features + class).
    Column index 13 has mostly '?' to exercise drop logic.
    Classes 1,2,3 with imbalance.
    """
    rng = np.random.RandomState(42)
    n = 40
    n_features = 15  # indices 0..14, target is col 15 → total 16 cols
    rows = []
    # Imbalanced labels: mostly 1, some 2, few 3
    labels = [1] * 24 + [2] * 10 + [3] * 6
    rng.shuffle(labels)
    for i in range(n):
        feats = rng.randn(n_features).tolist()
        # Encode some missing as '?'
        if i % 7 == 0:
            feats[10] = "?"
        # Column 13 heavily missing
        if i % 2 == 0:
            feats[13] = "?"
        rows.append(feats + [labels[i]])
    path = tmp_path / "tiny_arrhythmia.csv"
    pd.DataFrame(rows).to_csv(path, header=False, index=False)
    return path


@pytest.fixture
def synthetic_xy():
    """Imbalanced synthetic matrix suitable for ROS/SMOTE + PCA variants."""
    rng = np.random.RandomState(0)
    counts = {1: 40, 2: 15, 3: 8}
    X_parts = []
    y_parts = []
    for label, n in counts.items():
        X_parts.append(rng.randn(n, 12) + label)
        y_parts.append(np.full(n, label))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


@pytest.fixture
def temp_db(tmp_path: Path, schema_path: Path):
    from arrhythmia_ml.db import init_db

    db_path = tmp_path / "test_experiments.db"
    conn = init_db(db_path, schema_path=schema_path)
    yield conn, db_path
    conn.close()
