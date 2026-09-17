"""Load and preprocess the UCI Arrhythmia CSV (no train/test leakage)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

PathLike = Union[str, Path]

# Column 13 (UCI feature J) is dropped due to extreme missingness (~376/?).
DROP_COLUMN_INDEX = 13
IMPUTER_STRATEGY = "mean"


def _feature_names(n_features: int, drop_column_index: Optional[int]) -> list:
    """Build feature column names (without the target)."""
    if drop_column_index == 13:
        named = [
            "Age",
            "Sex",
            "Height",
            "Weight",
            "QRS_duration",
            "P-R_interval",
            "Q-T_interval",
            "T_interval",
            "P_interval",
            "QRS",
            "T",
            "P",
            "QRST",
            "Heart_rate",
        ]
    else:
        named = [
            "Age",
            "Sex",
            "Height",
            "Weight",
            "QRS_duration",
            "P-R_interval",
            "Q-T_interval",
            "T_interval",
            "P_interval",
            "QRS",
            "T",
            "P",
            "QRST",
            "J",
            "Heart_rate",
        ]
    names = []
    for i in range(n_features):
        if i < len(named):
            names.append(named[i])
        else:
            names.append(f"V{i}")
    return names


def load_raw_csv(csv_path: PathLike) -> pd.DataFrame:
    """Load arrhythmia CSV with no header."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, header=None)


def preprocess(
    csv_path: PathLike,
    *,
    drop_column_index: int = DROP_COLUMN_INDEX,
    imputer_strategy: str = IMPUTER_STRATEGY,
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Load arrhythmia data without fitting an imputer.

    Steps:
      1. Replace '?' with NaN
      2. Drop high-missingness column (default index 13 / J)
      3. Split features X and target y ('class')

    Missing values in X are left as NaN. Call ``impute_train_test`` after
    train/test splitting so SimpleImputer is fit on training data only.
    """
    df = load_raw_csv(csv_path)
    n_samples, n_cols_raw = df.shape

    df = df.replace("?", np.nan)
    if drop_column_index is not None:
        df = df.drop(columns=drop_column_index)

    df.columns = range(df.shape[1])
    # Coerce feature columns to numeric; target is last column.
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[df.columns[-1]] = pd.to_numeric(df[df.columns[-1]], errors="coerce")

    feature_names = _feature_names(df.shape[1] - 1, drop_column_index)
    df.columns = feature_names + ["class"]

    y = df["class"].astype(int)
    X = df.drop(columns=["class"])

    meta = {
        "dataset_path": str(csv_path),
        "n_samples": int(n_samples),
        "n_features_raw": int(n_cols_raw - 1),
        "n_features": int(X.shape[1]),
        "imputer_strategy": imputer_strategy,
        "drop_column_index": drop_column_index,
        "imputer_fitted": False,
    }
    return X, y, meta


def impute_train_test(
    X_train,
    X_test,
    *,
    strategy: str = IMPUTER_STRATEGY,
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer]:
    """
    Fit SimpleImputer on X_train only; transform train and test with that fit.

    Returns
    -------
    X_train_imp, X_test_imp, fitted_imputer
    """
    imputer = SimpleImputer(strategy=strategy)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    return X_train_imp, X_test_imp, imputer
