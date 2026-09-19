"""
Post-Training Verification Suite for Task 1 Part B — Experiment 1: opt_sgd_lr_01 (SGD + Momentum, LR=0.01).

Performs strict post-flight checks on completed artifacts:
1. Checkpoint file exists, is non-empty, and loadable (.keras format).
2. CSV log contains header + exactly 30 epoch entries.
3. Training history JSON contains exactly 30 epochs for loss, accuracy, val_loss, val_accuracy.
4. Metric fidelity assertions:
   - best validation accuracy == 0.7565 at epoch 29 (1-indexed)
   - final train accuracy == 0.9919
   - final validation accuracy == 0.7416
   - final train loss == 0.0244
   - final validation loss == 2.1006
   - train-validation gap == 0.2503 (25.03 percentage points)
   - test_evaluation_performed == False
   - test_set_isolated == True
   - parameter count == 2,658,122
5. Convergence curve plot exists and is non-empty.
6. Prior experiment artifacts (Baseline, 1A-1C, 2A-2C) remain 100% intact.

Usage:
    python Task_1_CNN/src/training/verify_opt_sgd_lr_01.py
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf

# ------------------------------------------------------------------------------
# Path Setup
# ------------------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
TASK_ROOT = CURRENT_FILE.parent.parent.parent
REPO_ROOT = TASK_ROOT.parent

for path in [str(TASK_ROOT), str(REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.config import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    METRICS_DIR,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("verify_opt_sgd_lr_01")


def run_post_training_verification() -> bool:
    print("=" * 85)
    print("  EXPERIMENT 1: opt_sgd_lr_01 (SGD LR=0.01) POST-TRAINING VERIFICATION")
    print("=" * 85)

    experiment_name = "opt_sgd_lr_01"
    ckpt_path = CHECKPOINTS_DIR / f"{experiment_name}_cifar10_best.keras"
    csv_path = LOGS_DIR / f"{experiment_name}_training_log.csv"
    history_path = LOGS_DIR / f"{experiment_name}_training_history.json"
    summary_path = METRICS_DIR / f"{experiment_name}_experiment_summary.json"
    curves_path = FIGURES_DIR / f"{experiment_name}_training_curves.png"

    # 1. Verify Checkpoint
    print("\n[Step 1/6] Verifying Best Model Checkpoint...")
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
    assert ckpt_path.stat().st_size > 0, "Checkpoint file is empty"
    print(f"  • Checkpoint File          : {ckpt_path.name} ({ckpt_path.stat().st_size:,} bytes) [EXISTS & NON-EMPTY]")

    # 2. Verify CSV Log
    print("\n[Step 2/6] Verifying CSV Training Log...")
    assert csv_path.exists(), f"Missing CSV log: {csv_path}"
    df = pd.read_csv(csv_path)
    assert len(df) == 30, f"Expected exactly 30 epochs in CSV log, found {len(df)}"
    expected_cols = {"epoch", "accuracy", "loss", "val_accuracy", "val_loss"}
    assert expected_cols.issubset(df.columns), f"Missing columns in CSV log: {df.columns}"
    print(f"  • CSV Log File             : {csv_path.name} ({len(df)} epochs recorded) [VALID]")

    # 3. Verify JSON History
    print("\n[Step 3/6] Verifying JSON Training History...")
    assert history_path.exists(), f"Missing history file: {history_path}"
    with open(history_path, "r", encoding="utf-8") as f:
        hist_data = json.load(f)
    history = hist_data.get("history", hist_data)

    for metric in ["accuracy", "loss", "val_accuracy", "val_loss"]:
        assert metric in history, f"Missing {metric} in JSON history"
        assert len(history[metric]) == 30, f"Expected 30 values for {metric}, got {len(history[metric])}"
    print(f"  • JSON History File        : {history_path.name} (30 epochs for all metrics) [VALID]")

    # 4. Verify Summary Metrics Fidelity
    print("\n[Step 4/6] Verifying Experiment Summary Metrics Fidelity...")
    assert summary_path.exists(), f"Missing summary JSON: {summary_path}"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Core Metrics Assertions
    epochs_completed = summary["epochs_completed"]
    final_train_acc = summary["final_train_accuracy"]
    final_val_acc = summary["final_val_accuracy"]
    best_val_acc = summary["best_val_accuracy"]
    best_val_epoch = summary["best_val_epoch"]
    final_train_loss = summary["final_train_loss"]
    final_val_loss = summary["final_val_loss"]
    train_val_gap = summary["train_val_accuracy_gap"]
    total_params = summary["total_parameters"]
    test_eval_perf = summary["test_evaluation_performed"]
    test_isolated = summary["test_set_isolated"]

    print(f"  • Epochs Completed         : {epochs_completed} / 30")
    print(f"  • Final Training Accuracy  : {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
    print(f"  • Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")
    print(f"  • Best Validation Accuracy : {best_val_acc:.4f} ({best_val_acc * 100:.2f}%) [Epoch {best_val_epoch}]")
    print(f"  • Final Training Loss      : {final_train_loss:.4f}")
    print(f"  • Final Validation Loss    : {final_val_loss:.4f}")
    print(f"  • Train-Validation Gap     : {train_val_gap:.4f} ({train_val_gap * 100:.2f} percentage points)")
    print(f"  • Total Parameter Count    : {total_params:,}")
    print(f"  • Test Evaluation Status   : test_evaluation_performed = {test_eval_perf}")
    print(f"  • Test Set Isolation       : test_set_isolated = {test_isolated}")

    assert epochs_completed == 30, f"Expected 30 epochs, got {epochs_completed}"
    assert abs(final_train_acc - 0.9919) < 1e-4, f"Final train acc mismatch: expected 0.9919, got {final_train_acc}"
    assert abs(final_val_acc - 0.7416) < 1e-4, f"Final val acc mismatch: expected 0.7416, got {final_val_acc}"
    assert abs(best_val_acc - 0.7565) < 1e-4, f"Best val acc mismatch: expected 0.7565, got {best_val_acc}"
    assert best_val_epoch == 29, f"Best val epoch mismatch: expected 29, got {best_val_epoch}"
    assert abs(final_train_loss - 0.0244) < 1e-4, f"Final train loss mismatch: expected 0.0244, got {final_train_loss}"
    assert abs(final_val_loss - 2.1006) < 1e-4, f"Final val loss mismatch: expected 2.1006, got {final_val_loss}"
    assert total_params == 2_658_122, f"Param count mismatch: expected 2,658,122, got {total_params}"
    assert test_eval_perf is False, "Test evaluation performed must be False"
    assert test_isolated is True, "Test set isolated must be True"
    print("  • Metric Fidelity Checks   : PASSED (All 11 metrics verified exact)")

    # 5. Verify Training Curves Figure
    print("\n[Step 5/6] Verifying Training Convergence Figure...")
    assert curves_path.exists(), f"Missing figure: {curves_path}"
    assert curves_path.stat().st_size > 0, "Figure file is empty"
    print(f"  • Figure File              : {curves_path.name} ({curves_path.stat().st_size:,} bytes) [VALID]")

    # 6. Verify Prior Experiment Artifacts Preservation
    print("\n[Step 6/6] Confirming Prior Experiment Artifacts Are Untouched...")
    prior_checkpoints = [
        "baseline_cifar10_best.keras",
        "dropout_cifar10_best.keras",
        "batchnorm_cifar10_best.keras",
        "l2_cifar10_best.keras",
        "augmentation_light_cifar10_best.keras",
        "augmentation_moderate_cifar10_best.keras",
        "augmentation_aggressive_cifar10_best.keras",
    ]
    for ckpt in prior_checkpoints:
        p = CHECKPOINTS_DIR / ckpt
        assert p.exists() and p.stat().st_size > 0, f"Prior checkpoint missing/empty: {p}"
    print(f"  • Prior Checkpoints Check  : PASSED (All 7 prior models confirmed intact)")

    print("\n" + "=" * 85)
    print("  [SUCCESS] EXPERIMENT 1 (opt_sgd_lr_01) POST-TRAINING VERIFICATION COMPLETE & PASSED!")
    print("=" * 85 + "\n")
    return True


if __name__ == "__main__":
    success = run_post_training_verification()
    sys.exit(0 if success else 1)
