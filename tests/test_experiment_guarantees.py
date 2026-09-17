"""Train-only oversampling and reproducibility guarantees."""

from __future__ import annotations

from arrhythmia_ml.config import ALL_VARIANTS, VARIANT_LABELS
from arrhythmia_ml.pipeline import run_variant


def test_variants_are_distinctly_labeled(synthetic_xy):
    X, y = synthetic_xy
    seen = {}
    for variant in ALL_VARIANTS:
        outcome = run_variant(X, y, variant=variant, drop_column_index=None)
        cfg = outcome["config"]
        assert cfg["variant"] == variant
        assert cfg["variant_label"] == VARIANT_LABELS[variant]
        seen[variant] = (
            cfg["oversampler"],
            cfg["pca_variance"],
            cfg["oversample_scope"],
        )

    assert seen["baseline"] == (None, None, None)
    assert seen["pca"][0] is None and seen["pca"][1] == 0.98
    assert seen["ros_pca"] == ("RandomOverSampler", 0.98, "train_only")
    assert seen["smote_pca"] == ("SMOTE", 0.98, "train_only")


def test_oversampling_does_not_change_test_size(synthetic_xy):
    X, y = synthetic_xy
    for variant in ("ros_pca", "smote_pca"):
        outcome = run_variant(X, y, variant=variant, drop_column_index=None)
        cfg = outcome["config"]
        assert cfg["oversample_scope"] == "train_only"
        assert cfg["test_rows"] == outcome["config"]["test_rows"]
        # Train grows; test stays at stratified holdout size
        assert cfg["train_rows_after"] > cfg["train_rows_before"]
        baseline = run_variant(X, y, variant="baseline", drop_column_index=None)
        assert cfg["test_rows"] == baseline["config"]["test_rows"]


def test_same_seeds_reproduce_metrics(synthetic_xy):
    X, y = synthetic_xy
    a = run_variant(
        X,
        y,
        variant="smote_pca",
        drop_column_index=None,
        random_state_split=1,
        random_state_model=0,
    )
    b = run_variant(
        X,
        y,
        variant="smote_pca",
        drop_column_index=None,
        random_state_split=1,
        random_state_model=0,
    )
    for ma, mb in zip(a["models"], b["models"]):
        assert ma["model_name"] == mb["model_name"]
        assert ma["metrics"] == mb["metrics"]
