"""
Visualization and Plotting Utilities for CIFAR-10 CNN Model Evaluation.

This module provides high-quality visualization generators:
1. 10x10 Confusion Matrix Heatmaps (Raw counts and Normalized percentages)
2. Per-Class Performance Bar Charts (Precision, Recall, F1-Score)
3. Training History Convergence Curves (Loss & Accuracy over epochs)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless scripts/servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup module logger
logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[Union[int, float]]]],
    class_names: List[str],
    filepath: Optional[Union[str, Path]] = None,
    title: str = "CIFAR-10 Baseline CNN - Confusion Matrix",
    normalize: bool = False,
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
) -> plt.Figure:
    """
    Generate and save a 10x10 confusion matrix heatmap plot.

    Args:
        cm (Union[np.ndarray, List[List[Union[int, float]]]]): 10x10 confusion matrix array.
        class_names (List[str]): Names of the target classes.
        filepath (Optional[Union[str, Path]]): Destination image path. If None, does not save.
        title (str): Title for the heatmap plot.
        normalize (bool): If True, normalizes row values to percentages [0.0, 1.0]. Defaults to False.
        cmap (str): Seaborn colormap name. Defaults to "Blues".
        figsize (Tuple[int, int]): Figure dimensions (width, height) in inches. Defaults to (10, 8).
        dpi (int): Resolution in dots per inch. Defaults to 300.

    Returns:
        plt.Figure: The created Matplotlib Figure object.
    """
    cm_arr = np.array(cm, dtype=np.float64)

    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_display = np.nan_to_num(cm_arr / cm_arr.sum(axis=1, keepdims=True))
        fmt = ".2%"
        cbar_label = "Proportion / Recall"
        if "Normalized" not in title:
            title = f"{title} (Normalized)"
    else:
        cm_display = cm_arr.astype(int)
        fmt = "d"
        cbar_label = "Sample Count"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": cbar_label},
        linewidths=0.5,
        linecolor="#e0e0e0",
        square=True,
        ax=ax,
        annot_kws={"size": 9 if normalize else 10, "weight": "bold"},
    )

    ax.set_title(title, fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=12, weight="bold", labelpad=10)
    ax.set_ylabel("True Class", fontsize=12, weight="bold", labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()

    if filepath is not None:
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved confusion matrix plot to: {save_path}")

    plt.close(fig)
    return fig


def plot_per_class_metrics(
    metrics_dict: Dict[str, Any],
    filepath: Optional[Union[str, Path]] = None,
    title: str = "CIFAR-10 Baseline CNN - Per-Class Classification Performance",
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Generate and save a grouped bar chart of Precision, Recall, and F1-Score per class.

    Args:
        metrics_dict (Dict[str, Any]): Dictionary of metrics containing "per_class".
        filepath (Optional[Union[str, Path]]): Destination image path. If None, does not save.
        title (str): Plot title.
        figsize (Tuple[int, int]): Figure dimensions. Defaults to (12, 6).
        dpi (int): Image resolution. Defaults to 300.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    per_class = metrics_dict["per_class"]
    class_names = list(per_class.keys())
    precisions = [per_class[c]["precision"] for c in class_names]
    recalls = [per_class[c]["recall"] for c in class_names]
    f1s = [per_class[c]["f1_score"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    rects1 = ax.bar(x - width, precisions, width, label="Precision", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x, recalls, width, label="Recall", color="#4682b4", alpha=0.9)
    rects3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#87ceeb", edgecolor="#2b5c8f", alpha=0.95)

    ax.set_ylabel("Score", fontsize=12, weight="bold")
    ax.set_title(title, fontsize=14, weight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10, weight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add overall accuracy line
    if "overall_accuracy" in metrics_dict:
        acc = metrics_dict["overall_accuracy"]
        ax.axhline(y=acc, color="#d9534f", linestyle=":", linewidth=1.5, label=f"Overall Accuracy ({acc:.2%})")
        ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=10)

    plt.tight_layout()

    if filepath is not None:
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved per-class metrics bar chart to: {save_path}")

    plt.close(fig)
    return fig


def plot_training_history(
    history_source: Union[str, Path, Dict[str, Any]],
    filepath: Optional[Union[str, Path]] = None,
    title: str = "CIFAR-10 Baseline CNN - Training & Validation Convergence",
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 300,
) -> plt.Figure:
    """
    Generate and save training vs validation Loss and Accuracy curves over all epochs.

    Args:
        history_source (Union[str, Path, Dict[str, Any]]): JSON path, CSV path, or history dict.
        filepath (Optional[Union[str, Path]]): Destination image path. If None, does not save.
        title (str): Main plot title.
        figsize (Tuple[int, int]): Figure dimensions. Defaults to (14, 5).
        dpi (int): Image resolution. Defaults to 300.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    # Load history data
    if isinstance(history_source, (str, Path)):
        p = Path(history_source)
        if p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            history = data.get("history", data)
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
            history = {col: df[col].tolist() for col in df.columns}
        else:
            raise ValueError(f"Unsupported history file format: {p.suffix}")
    else:
        history = history_source.get("history", history_source)

    epochs = range(1, len(history.get("loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # 1. Loss Curve
    if "loss" in history:
        ax1.plot(epochs, history["loss"], "o-", label="Training Loss", color="#1f77b4", linewidth=2, markersize=4)
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], "s--", label="Validation Loss", color="#ff7f0e", linewidth=2, markersize=4)
        min_val_loss = min(history["val_loss"])
        min_epoch = history["val_loss"].index(min_val_loss) + 1
        ax1.plot(min_epoch, min_val_loss, "r*", markersize=12, label=f"Min Val Loss: {min_val_loss:.4f} (Ep {min_epoch})")

    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, weight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Accuracy Curve
    if "accuracy" in history:
        ax2.plot(epochs, history["accuracy"], "o-", label="Training Accuracy", color="#2ca02c", linewidth=2, markersize=4)
    if "val_accuracy" in history:
        ax2.plot(epochs, history["val_accuracy"], "s--", label="Validation Accuracy", color="#d62728", linewidth=2, markersize=4)
        max_val_acc = max(history["val_accuracy"])
        max_epoch = history["val_accuracy"].index(max_val_acc) + 1
        ax2.plot(max_epoch, max_val_acc, "r*", markersize=12, label=f"Max Val Acc: {max_val_acc:.4f} (Ep {max_epoch})")

    ax2.set_title("Accuracy vs. Epochs", fontsize=12, weight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()

    if filepath is not None:
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved training convergence curves to: {save_path}")

    plt.close(fig)
    return fig
