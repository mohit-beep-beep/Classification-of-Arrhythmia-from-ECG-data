"""Tests for experiment variants."""

from __future__ import annotations

from arrhythmia_ml.pipeline import ALL_VARIANTS, run_variant


def test_all_variants_produce_valid_metrics(synthetic_xy):
    X, y = synthetic_xy
    for variant in ALL_VARIANTS:
        outcome = run_variant(
            X,
            y,
            variant=variant,
            drop_column_index=None,
            test_size=0.25,
        )
        assert outcome["config"]["variant"] == variant
        assert len(outcome["models"]) == 6
        for model in outcome["models"]:
            for split in ("train", "test"):
                for name, value in model["metrics"][split].items():
                    assert 0.0 <= value <= 1.0, (variant, model["model_name"], name, value)


def test_oversampling_increases_train_rows(synthetic_xy):
    X, y = synthetic_xy
    baseline = run_variant(X, y, variant="baseline", drop_column_index=None)
    ros = run_variant(X, y, variant="ros_pca", drop_column_index=None)
    smote = run_variant(X, y, variant="smote_pca", drop_column_index=None)

    assert ros["config"]["train_rows_after"] > ros["config"]["train_rows_before"]
    assert smote["config"]["train_rows_after"] > smote["config"]["train_rows_before"]
    assert baseline["config"]["train_rows_after"] == baseline["config"]["train_rows_before"]
    assert baseline["config"]["oversampler"] is None
    assert ros["config"]["oversampler"] == "RandomOverSampler"
    assert smote["config"]["oversampler"] == "SMOTE"
    assert ros["config"]["oversample_scope"] == "train_only"


def test_pca_sets_n_components(synthetic_xy):
    X, y = synthetic_xy
    outcome = run_variant(X, y, variant="pca", drop_column_index=None)
    assert outcome["config"]["pca_variance"] == 0.98
    assert outcome["config"]["pca_n_components"] is not None
    assert outcome["config"]["pca_n_components"] >= 1
