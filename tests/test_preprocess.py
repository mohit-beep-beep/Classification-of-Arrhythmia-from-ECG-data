"""Tests for preprocessing (no full-dataset imputation)."""

from __future__ import annotations

import numpy as np

from arrhythmia_ml.preprocess import (
    DROP_COLUMN_INDEX,
    impute_train_test,
    preprocess,
)


def test_preprocess_loads_without_imputing(tiny_csv):
    X, y, meta = preprocess(tiny_csv)
    assert meta["drop_column_index"] == DROP_COLUMN_INDEX
    assert meta["imputer_strategy"] == "mean"
    assert meta["imputer_fitted"] is False
    # Original 15 features + class; drop col 13 → 14 features
    assert X.shape[1] == 14
    assert len(y) == X.shape[0] == 40
    # Missing values remain until train-only imputation
    assert X.isna().any().any()
    assert not y.isna().any()
    assert set(y.unique()).issubset({1, 2, 3})


def test_preprocess_missing_file(tmp_path):
    missing = tmp_path / "nope.csv"
    try:
        preprocess(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_imputer_fitted_only_on_training_rows():
    """
    Imputer statistics must equal train-column means, not full-data means.
    """
    # Explicit partitions so train/full means differ sharply on column 0.
    X_train = np.array(
        [
            [0.0, 1.0],
            [2.0, np.nan],
            [4.0, 3.0],
            [np.nan, 5.0],
        ],
        dtype=float,
    )
    X_test = np.array(
        [
            [100.0, np.nan],
            [np.nan, 200.0],
            [300.0, 400.0],
        ],
        dtype=float,
    )
    X_full = np.vstack([X_train, X_test])

    full_mean_col0 = np.nanmean(X_full[:, 0])
    train_mean_col0 = np.nanmean(X_train[:, 0])  # (0+2+4)/3 = 2.0
    assert abs(full_mean_col0 - train_mean_col0) > 50.0

    X_train_i, X_test_i, imputer = impute_train_test(X_train, X_test, strategy="mean")

    assert np.isfinite(X_train_i).all()
    assert np.isfinite(X_test_i).all()
    np.testing.assert_allclose(imputer.statistics_[0], train_mean_col0)
    assert not np.isclose(imputer.statistics_[0], full_mean_col0)
    # Test NaNs filled with train mean (2.0), not test/full mean
    assert X_test_i[1, 0] == np.float64(2.0) or np.isclose(X_test_i[1, 0], 2.0)


def test_impute_train_test_does_not_use_test_values_in_fit():
    """Changing test values must not change imputer.statistics_."""
    X_train = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    X_test_a = np.array([[np.nan, 100.0], [np.nan, 200.0]])
    X_test_b = np.array([[np.nan, -999.0], [np.nan, -888.0]])

    _, _, imp_a = impute_train_test(X_train, X_test_a, strategy="mean")
    _, _, imp_b = impute_train_test(X_train, X_test_b, strategy="mean")

    np.testing.assert_allclose(imp_a.statistics_, imp_b.statistics_)
    # Train-only means: col0 = (1+3+5)/3 = 3, col1 = (2+6)/2 = 4
    np.testing.assert_allclose(imp_a.statistics_, [3.0, 4.0])
