"""Classifier factory matching the six notebook models."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def get_classifiers(random_state: int = 0) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """
    Return (model_name, estimator, hyperparameters_dict) for the six classifiers.

    Hyperparameters are aligned with the consolidated notebook defaults.
    """
    specs = [
        (
            "KNN Classifier",
            KNeighborsClassifier(n_neighbors=5),
            {"n_neighbors": 5},
        ),
        (
            "Logistic Regression",
            LogisticRegression(
                solver="saga",
                random_state=random_state,
                max_iter=5000,
            ),
            {
                "solver": "saga",
                "random_state": random_state,
                "max_iter": 5000,
            },
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(
                criterion="entropy",
                max_depth=5,
                random_state=random_state,
            ),
            {
                "criterion": "entropy",
                "max_depth": 5,
                "random_state": random_state,
            },
        ),
        (
            "Linear SVC",
            LinearSVC(
                C=0.01,
                random_state=random_state,
                max_iter=10000,
            ),
            {
                "C": 0.01,
                "random_state": random_state,
                "max_iter": 10000,
            },
        ),
        (
            "Kernelized SVC",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma=0.1,
                random_state=random_state,
            ),
            {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": 0.1,
                "random_state": random_state,
            },
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                criterion="gini",
                max_features=None,
                max_depth=10,
                max_leaf_nodes=30,
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "n_estimators": 300,
                "criterion": "gini",
                "max_depth": 10,
                "max_leaf_nodes": 30,
                "random_state": random_state,
            },
        ),
    ]
    return specs
