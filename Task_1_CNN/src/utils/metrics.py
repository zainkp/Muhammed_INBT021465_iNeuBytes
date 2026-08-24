"""
Quantitative Evaluation Metrics Utility Module for CIFAR-10 Classification.

This module provides comprehensive metric calculation, formatting, and serialization
utilities for multiclass classification tasks:
- Overall accuracy, test loss, sample counts, and class counts
- Per-class Precision, Recall, F1-Score, and Support
- Macro-averaged and Weighted-averaged Precision, Recall, and F1-Score
- 10x10 Raw and Normalized Confusion Matrices
- Tabular console reporting
- Structured JSON and CSV metric export utilities
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Setup module logger
logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    digits: int = 4,
) -> Dict[str, Any]:
    """
    Compute overall and per-class classification metrics on predictions vs ground truth.

    Args:
        y_true (np.ndarray): 1D array of true integer labels, shape (N,).
        y_pred (np.ndarray): 1D array of predicted integer labels, shape (N,).
        class_names (Optional[List[str]]): List of class label names. Defaults to indices [0..C-1].
        digits (int): Decimal precision for rounding metrics. Defaults to 4.

    Returns:
        Dict[str, Any]: Nested dictionary containing:
            - "overall_accuracy": float
            - "total_samples": int
            - "num_classes": int
            - "macro_avg": {"precision": float, "recall": float, "f1_score": float}
            - "weighted_avg": {"precision": float, "recall": float, "f1_score": float}
            - "per_class": {class_name: {"precision": float, "recall": float, "f1_score": float, "support": int}}
            - "confusion_matrix": List[List[int]] (raw counts)
            - "confusion_matrix_normalized": List[List[float]] (row-normalized percentages)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)} samples."
        )

    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(class_names) if class_names is not None else len(unique_classes)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Overall accuracy
    acc = float(accuracy_score(y_true, y_pred))

    # Per-class metrics
    p_per_class = precision_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    r_per_class = recall_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)

    # Class supports
    supports = [int(np.sum(y_true == i)) for i in range(num_classes)]

    per_class_dict: Dict[str, Dict[str, Union[float, int]]] = {}
    for i, name in enumerate(class_names):
        per_class_dict[name] = {
            "class_id": i,
            "precision": round(float(p_per_class[i]), digits),
            "recall": round(float(r_per_class[i]), digits),
            "f1_score": round(float(f1_per_class[i]), digits),
            "support": supports[i],
        }

    # Macro and Weighted Averages
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion matrix
    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    # Row-normalized confusion matrix (recall per cell)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm_raw.astype(np.float64) / cm_raw.sum(axis=1, keepdims=True))

    metrics_result: Dict[str, Any] = {
        "total_samples": int(len(y_true)),
        "num_classes": num_classes,
        "class_names": class_names,
        "overall_accuracy": round(acc, digits),
        "overall_accuracy_percent": round(acc * 100.0, 2),
        "macro_avg": {
            "precision": round(macro_p, digits),
            "recall": round(macro_r, digits),
            "f1_score": round(macro_f1, digits),
        },
        "weighted_avg": {
            "precision": round(weighted_p, digits),
            "recall": round(weighted_r, digits),
            "f1_score": round(weighted_f1, digits),
        },
        "per_class": per_class_dict,
        "confusion_matrix": cm_raw.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, digits).tolist(),
    }

    return metrics_result


def format_metrics_table(metrics_dict: Dict[str, Any]) -> str:
    """
    Format per-class and summary metrics into an aligned human-readable text table.

    Args:
        metrics_dict (Dict[str, Any]): Dictionary output from compute_classification_metrics.

    Returns:
        str: Formatted markdown-style text table.
    """
    lines = []
    header = f"{'Class ID':<10} | {'Class Name':<14} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11} | {'Support':<8}"
    divider = "-" * len(header)
    lines.append(divider)
    lines.append(header)
    lines.append(divider)

    for class_name, metrics in metrics_dict["per_class"].items():
        cid = metrics["class_id"]
        p = metrics["precision"]
        r = metrics["recall"]
        f1 = metrics["f1_score"]
        sup = metrics["support"]
        lines.append(f"{cid:<10} | {class_name:<14} | {p:<11.4f} | {r:<11.4f} | {f1:<11.4f} | {sup:<8,}")

    lines.append(divider)
    macro = metrics_dict["macro_avg"]
    lines.append(
        f"{'':<10} | {'macro avg':<14} | {macro['precision']:<11.4f} | {macro['recall']:<11.4f} | {macro['f1_score']:<11.4f} | {metrics_dict['total_samples']:<8,}"
    )
    weighted = metrics_dict["weighted_avg"]
    lines.append(
        f"{'':<10} | {'weighted avg':<14} | {weighted['precision']:<11.4f} | {weighted['recall']:<11.4f} | {weighted['f1_score']:<11.4f} | {metrics_dict['total_samples']:<8,}"
    )
    acc = metrics_dict["overall_accuracy"]
    lines.append(
        f"{'':<10} | {'accuracy':<14} | {'':<11} | {'':<11} | {acc:<11.4f} | {metrics_dict['total_samples']:<8,}"
    )
    lines.append(divider)

    return "\n".join(lines)


def save_metrics_json(metrics_dict: Dict[str, Any], filepath: Union[str, Path]) -> Path:
    """
    Serialize and save complete metrics dictionary to JSON.

    Args:
        metrics_dict (Dict[str, Any]): Dictionary of computed metrics.
        filepath (Union[str, Path]): Target output filepath.

    Returns:
        Path: Destination path.
    """
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Saved evaluation metrics JSON to: {save_path}")
    return save_path


def save_classification_report_csv(
    metrics_dict: Dict[str, Any],
    filepath: Union[str, Path],
) -> Path:
    """
    Export classification metrics (per-class, macro, weighted, accuracy) to a tidy CSV file.

    Args:
        metrics_dict (Dict[str, Any]): Dictionary of computed metrics.
        filepath (Union[str, Path]): Target CSV filepath.

    Returns:
        Path: Destination path.
    """
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for class_name, metrics in metrics_dict["per_class"].items():
        rows.append({
            "class_id": metrics["class_id"],
            "class_name": class_name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "support": metrics["support"],
        })

    # Summary rows
    rows.append({
        "class_id": "",
        "class_name": "accuracy",
        "precision": "",
        "recall": "",
        "f1_score": metrics_dict["overall_accuracy"],
        "support": metrics_dict["total_samples"],
    })
    rows.append({
        "class_id": "",
        "class_name": "macro avg",
        "precision": metrics_dict["macro_avg"]["precision"],
        "recall": metrics_dict["macro_avg"]["recall"],
        "f1_score": metrics_dict["macro_avg"]["f1_score"],
        "support": metrics_dict["total_samples"],
    })
    rows.append({
        "class_id": "",
        "class_name": "weighted avg",
        "precision": metrics_dict["weighted_avg"]["precision"],
        "recall": metrics_dict["weighted_avg"]["recall"],
        "f1_score": metrics_dict["weighted_avg"]["f1_score"],
        "support": metrics_dict["total_samples"],
    })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved classification report CSV to: {save_path}")
    return save_path


def save_confusion_matrix_csv(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: List[str],
    filepath: Union[str, Path],
) -> Path:
    """
    Export 10x10 confusion matrix to a labeled CSV file.

    Args:
        cm (Union[np.ndarray, List[List[int]]]): Confusion matrix.
        class_names (List[str]): List of class label names.
        filepath (Union[str, Path]): Target CSV filepath.

    Returns:
        Path: Destination path.
    """
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.asarray(cm)
    df = pd.DataFrame(cm_arr, index=class_names, columns=class_names)
    df.index.name = "true_class"
    df.to_csv(save_path)
    logger.info(f"Saved confusion matrix CSV to: {save_path}")
    return save_path
