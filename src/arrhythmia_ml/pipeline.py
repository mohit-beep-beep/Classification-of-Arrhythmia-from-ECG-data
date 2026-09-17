"""Experiment pipeline variants: baseline, pca, ros_pca, smote_pca."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    ALL_VARIANTS,
    DROP_COLUMN_INDEX,
    IMPUTER_STRATEGY,
    PCA_VARIANCE,
    RANDOM_STATE_MODEL,
    RANDOM_STATE_SPLIT,
    TEST_SIZE,
    VARIANT_BASELINE,
    VARIANT_LABELS,
    VARIANT_PCA,
    VARIANT_ROS_PCA,
    VARIANT_SMOTE_PCA,
)
from .evaluate import compute_split_metrics, flatten_confusion_matrix
from .models import get_classifiers
from .preprocess import impute_train_test

__all__ = [
    "ALL_VARIANTS",
    "VARIANT_BASELINE",
    "VARIANT_PCA",
    "VARIANT_ROS_PCA",
    "VARIANT_SMOTE_PCA",
    "VARIANT_LABELS",
    "run_variant",
    "run_all_variants",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _class_counts(y) -> Dict[int, int]:
    return {int(k): int(v) for k, v in Counter(np.asarray(y)).items()}


def _variant_settings(variant: str) -> Dict[str, Any]:
    if variant == VARIANT_BASELINE:
        return {
            "use_pca": False,
            "pca_variance": None,
            "oversampler": None,
            "oversample_scope": None,
        }
    if variant == VARIANT_PCA:
        return {
            "use_pca": True,
            "pca_variance": PCA_VARIANCE,
            "oversampler": None,
            "oversample_scope": None,
        }
    if variant == VARIANT_ROS_PCA:
        return {
            "use_pca": True,
            "pca_variance": PCA_VARIANCE,
            "oversampler": "RandomOverSampler",
            "oversample_scope": "train_only",
        }
    if variant == VARIANT_SMOTE_PCA:
        return {
            "use_pca": True,
            "pca_variance": PCA_VARIANCE,
            "oversampler": "SMOTE",
            "oversample_scope": "train_only",
        }
    raise ValueError(f"Unknown variant: {variant}. Expected one of {ALL_VARIANTS}")


def _apply_pca(X_train_s, X_test_s, pca_variance: float, random_state: int):
    """Fit PCA on train only; transform train and test."""
    n_features = X_train_s.shape[1]
    n_samples = X_train_s.shape[0]
    max_components = min(n_samples, n_features)

    if max_components <= 1:
        pca = PCA(n_components=1, random_state=random_state)
        return pca.fit_transform(X_train_s), pca.transform(X_test_s), pca

    try:
        pca = PCA(n_components=pca_variance, random_state=random_state)
        X_tr = pca.fit_transform(X_train_s)
        X_te = pca.transform(X_test_s)
        return X_tr, X_te, pca
    except ValueError:
        pca = PCA(
            n_components=max(1, max_components - 1),
            random_state=random_state,
        )
        return pca.fit_transform(X_train_s), pca.transform(X_test_s), pca


def run_variant(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    *,
    variant: str,
    test_size: float = TEST_SIZE,
    random_state_split: int = RANDOM_STATE_SPLIT,
    random_state_model: int = RANDOM_STATE_MODEL,
    imputer_strategy: str = IMPUTER_STRATEGY,
    drop_column_index: Optional[int] = DROP_COLUMN_INDEX,
    stratify: bool = True,
) -> Dict[str, Any]:
    """
    Run one experiment variant and return structured results for DB persistence.

    Order (leak-safe):
      1. Train/test split
      2. SimpleImputer fit on X_train only, then transform train + test
      3. Optional ROS/SMOTE on train only
      4. StandardScaler / optional PCA fit on train only
      5. Train classifiers; evaluate with sklearn metrics
    """
    settings = _variant_settings(variant)

    y_arr = np.asarray(y)
    X_arr = np.asarray(X, dtype=float)

    stratify_y = y_arr if stratify else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr,
            y_arr,
            test_size=test_size,
            random_state=random_state_split,
            stratify=stratify_y,
        )
    except ValueError:
        # Too few samples per class for stratify (tiny fixtures).
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr,
            y_arr,
            test_size=test_size,
            random_state=random_state_split,
            stratify=None,
        )

    # Impute after split: fit ONLY on training features.
    X_train, X_test, _imputer = impute_train_test(
        X_train, X_test, strategy=imputer_strategy
    )

    # Freeze test set before any oversampling — ROS/SMOTE must not touch it.
    X_test_frozen = np.array(X_test, copy=True)
    y_test_frozen = np.array(y_test, copy=True)
    test_rows = int(len(y_test_frozen))

    distributions: Dict[str, Dict[int, int]] = {
        "raw": _class_counts(y_arr),
        "train_before": _class_counts(y_train),
        "test": _class_counts(y_test_frozen),
    }
    train_rows_before = int(len(y_train))

    if settings["oversampler"] == "RandomOverSampler":
        sampler = RandomOverSampler(random_state=random_state_model)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    elif settings["oversampler"] == "SMOTE":
        min_count = int(min(Counter(y_train).values()))
        k = max(1, min(5, min_count - 1)) if min_count > 1 else 1
        sampler = SMOTE(random_state=random_state_model, k_neighbors=k)
        X_train, y_train = sampler.fit_resample(X_train, y_train)

    # Guard: test set must be unchanged after oversampling step.
    if not np.array_equal(X_test, X_test_frozen) or not np.array_equal(
        y_test, y_test_frozen
    ):
        raise RuntimeError("Test set was modified during oversampling (leakage).")

    distributions["train_after"] = _class_counts(y_train)
    train_rows_after = int(len(y_train))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test_frozen)

    pca_n_components = None
    if settings["use_pca"]:
        X_train_s, X_test_s, pca = _apply_pca(
            X_train_s,
            X_test_s,
            settings["pca_variance"],
            random_state_model,
        )
        pca_n_components = int(pca.n_components_)

    config = {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "test_size": float(test_size),
        "random_state_split": int(random_state_split),
        "random_state_model": int(random_state_model),
        "imputer_strategy": imputer_strategy,
        "drop_column_index": drop_column_index,
        "scaler": "StandardScaler",
        "pca_variance": settings["pca_variance"],
        "pca_n_components": pca_n_components,
        "oversampler": settings["oversampler"],
        "oversample_scope": settings["oversample_scope"],
        "train_rows_before": train_rows_before,
        "train_rows_after": train_rows_after,
        "test_rows": test_rows,
    }

    models_out: List[Dict[str, Any]] = []
    for model_name, estimator, hyperparams in get_classifiers(
        random_state=random_state_model
    ):
        clf = estimator
        clf.fit(X_train_s, y_train)

        y_train_pred = clf.predict(X_train_s)
        y_test_pred = clf.predict(X_test_s)

        metrics = {
            "train": compute_split_metrics(y_train, y_train_pred),
            "test": compute_split_metrics(y_test_frozen, y_test_pred),
        }
        models_out.append(
            {
                "model_name": model_name,
                "sklearn_class": f"{type(clf).__module__}.{type(clf).__name__}",
                "hyperparameters": hyperparams,
                "trained_at": _utc_now(),
                "metrics": metrics,
                "cm_rows": flatten_confusion_matrix(y_test_frozen, y_test_pred),
                "y_test": np.asarray(y_test_frozen),
                "y_test_pred": np.asarray(y_test_pred),
            }
        )

    return {
        "config": config,
        "distributions": distributions,
        "models": models_out,
    }


def run_all_variants(
    X,
    y,
    *,
    variants: Sequence[str] = ALL_VARIANTS,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Run multiple variants sequentially."""
    return [run_variant(X, y, variant=v, **kwargs) for v in variants]
