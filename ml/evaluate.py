"""
ml/evaluate.py
===============
Shared evaluation utilities for both regression and classification models.

Regression metrics  : MAE, RMSE, R²
Classification metrics : Accuracy, Precision, Recall, F1 (weighted),
                         Confusion matrix, per-class report
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from config.logger import get_logger

logger = get_logger(__name__, log_file="ml.log")


# ── Regression ────────────────────────────────────────────────────────────────

def evaluate_regression(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    split_name: str = "test",
) -> dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Parameters
    ----------
    y_true     : Ground-truth AQI values
    y_pred     : Predicted AQI values
    model_name : Label for logging
    split_name : "train" | "val" | "test"

    Returns
    -------
    dict with keys: mae, rmse, r2
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}
    logger.info(
        "Regression [%s] %s → MAE=%.4f | RMSE=%.4f | R²=%.4f",
        model_name, split_name, mae, rmse, r2,
    )
    return metrics


# ── Classification ────────────────────────────────────────────────────────────

def evaluate_classification(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[list[str]] = None,
    model_name: str = "",
    split_name: str = "test",
) -> dict:
    """
    Calculate classification evaluation metrics.

    Parameters
    ----------
    y_true      : Ground-truth encoded labels (int)
    y_pred      : Predicted encoded labels (int)
    label_names : Human-readable class names (from LabelEncoder.classes_)
    model_name  : Label for logging
    split_name  : "train" | "val" | "test"

    Returns
    -------
    dict with keys: accuracy, precision_weighted, recall_weighted,
                    f1_weighted, confusion_matrix, classification_report
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "accuracy":             round(acc,  4),
        "precision_weighted":   round(prec, 4),
        "recall_weighted":      round(rec,  4),
        "f1_weighted":          round(f1,   4),
        "confusion_matrix":     cm.tolist(),
        "classification_report": report,
    }

    logger.info(
        "Classification [%s] %s → Acc=%.4f | P=%.4f | R=%.4f | F1=%.4f",
        model_name, split_name, acc, prec, rec, f1,
    )
    return metrics


def print_regression_summary(results: dict[str, dict]) -> None:
    """Print a formatted comparison table for multiple regression models."""
    print("\n" + "=" * 65)
    print(f"{'Model':<35} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("=" * 65)
    for name, m in sorted(results.items(), key=lambda x: x[1].get("test_mae", 9999)):
        print(
            f"{name:<35} "
            f"{m.get('test_mae', 'N/A'):>8.4f} "
            f"{m.get('test_rmse', 'N/A'):>8.4f} "
            f"{m.get('test_r2', 'N/A'):>8.4f}"
        )
    print("=" * 65)


def print_classification_summary(results: dict[str, dict]) -> None:
    """Print a formatted comparison table for multiple classification models."""
    print("\n" + "=" * 75)
    print(f"{'Model':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("=" * 75)
    for name, m in sorted(results.items(), key=lambda x: x[1].get("test_f1_weighted", 0), reverse=True):
        print(
            f"{name:<35} "
            f"{m.get('test_accuracy', 'N/A'):>10.4f} "
            f"{m.get('test_precision_weighted', 'N/A'):>10.4f} "
            f"{m.get('test_recall_weighted', 'N/A'):>10.4f} "
            f"{m.get('test_f1_weighted', 'N/A'):>10.4f}"
        )
    print("=" * 75)
