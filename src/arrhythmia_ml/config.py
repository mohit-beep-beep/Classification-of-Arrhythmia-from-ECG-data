"""Shared reproducible defaults for arrhythmia experiments."""

# Fixed seeds (aligned with original notebooks where applicable).
RANDOM_STATE_SPLIT = 1
RANDOM_STATE_MODEL = 0
TEST_SIZE = 0.2
PCA_VARIANCE = 0.98
DROP_COLUMN_INDEX = 13
IMPUTER_STRATEGY = "mean"

VARIANT_BASELINE = "baseline"
VARIANT_PCA = "pca"
VARIANT_ROS_PCA = "ros_pca"
VARIANT_SMOTE_PCA = "smote_pca"

ALL_VARIANTS = (
    VARIANT_BASELINE,
    VARIANT_PCA,
    VARIANT_ROS_PCA,
    VARIANT_SMOTE_PCA,
)

# Human-readable labels for analytics / dashboards.
VARIANT_LABELS = {
    VARIANT_BASELINE: "Baseline (no PCA, no oversampling)",
    VARIANT_PCA: "PCA only (98% variance)",
    VARIANT_ROS_PCA: "ROS + PCA (train-only RandomOverSampler)",
    VARIANT_SMOTE_PCA: "SMOTE + PCA (train-only SMOTE)",
}
