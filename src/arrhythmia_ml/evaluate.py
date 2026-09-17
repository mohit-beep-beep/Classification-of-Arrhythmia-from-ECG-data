"""Evaluation helpers — metrics always computed from real predictions."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_split_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute scalar classification metrics for one split."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def flatten_confusion_matrix(y_true, y_pred) -> List[Tuple[int, int, int]]:
    """
    Return non-zero confusion matrix cells as (true_label, pred_label, count).
    Labels are the sorted unique labels present in y_true ∪ y_pred.
    """
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows: List[Tuple[int, int, int]] = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            count = int(cm[i, j])
            if count > 0:
                rows.append((int(true_label), int(pred_label), count))
    return rows
