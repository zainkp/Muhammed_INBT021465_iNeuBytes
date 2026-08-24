"""
Execution Entry Point for Baseline CNN Test-Set Evaluation on CIFAR-10.

This script executes the complete final test evaluation stage for Task 1:
1. Sets deterministic random seeds across Python, NumPy, and TensorFlow.
2. Loads CIFAR-10 test set using the validated frozen split pipeline (10,000 images).
3. Strictly isolates the test set: training and validation splits are discarded.
4. Loads the best baseline model checkpoint (models/checkpoints/baseline_cifar10_best.keras).
5. Evaluates model predictions over the entire 10,000-sample test set.
6. Computes Test Loss, Test Accuracy, Sample Count, Class Count.
7. Calculates Per-Class Precision, Recall, F1-Score, Support, and Macro/Weighted averages.
8. Generates a 10x10 Confusion Matrix (raw counts and normalized proportions).
9. Persists evaluation artifacts into results/metrics/ and figures/.
10. Prints a clear, comprehensive, formatted evaluation summary to the console.

Usage:
    Direct execution:
        python Task_1_CNN/src/training/run_evaluation.py

    Module execution (from Task_1_CNN directory):
        python -m src.training.run_evaluation

    Module execution (from workspace root):
        python -m Task_1_CNN.src.training.run_evaluation
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# ------------------------------------------------------------------------------
# Path Setup (Supports module execution, direct script execution, and repo-level CLI)
# ------------------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
TASK_ROOT = CURRENT_FILE.parent.parent.parent
REPO_ROOT = TASK_ROOT.parent

for path in [str(TASK_ROOT), str(REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# ------------------------------------------------------------------------------
# Source Module Imports (Reuse existing codebase components without duplication)
# ------------------------------------------------------------------------------
from src.config import (
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    FIGURES_DIR,
    LOGS_DIR,
    METRICS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
    SPLITS_DIR,
    TEST_SAMPLE_COUNT,
)
from src.data.dataset import load_cifar10_data
from src.training.evaluator import BaselineEvaluator
from src.utils.metrics import format_metrics_table
from src.utils.seed import set_seed

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_evaluation")


def run_baseline_evaluation(
    checkpoint_filename: str = "baseline_cifar10_best.keras",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Execute end-to-end evaluation of the best baseline CNN checkpoint on the test set.

    Args:
        checkpoint_filename (str): Name of the best checkpoint file in models/checkpoints/.
        batch_size (int): Batch size for inference evaluation.

    Returns:
        Dict[str, Any]: Complete metrics dictionary.
    """
    print("\n" + "=" * 80)
    print("        CIFAR-10 BASELINE CNN: FINAL TEST SET EVALUATION")
    print("=" * 80)

    # 1. Reproducibility
    logger.info(f"Setting global random seed: {RANDOM_SEED}")
    set_seed(RANDOM_SEED)

    # 2. Checkpoint Verification
    checkpoint_filepath = Path(CHECKPOINTS_DIR) / checkpoint_filename
    if not checkpoint_filepath.exists():
        raise FileNotFoundError(
            f"Required baseline checkpoint not found at: {checkpoint_filepath}\n"
            f"Please ensure the 30-epoch baseline training run has completed successfully."
        )

    print(f"\n[Model Checkpoint Source]")
    print(f"  • Filepath : {checkpoint_filepath}")
    print(f"  • Size     : {checkpoint_filepath.stat().st_size / (1024 * 1024):.2f} MB")

    # 3. Load CIFAR-10 Data via Frozen Split Pipeline
    logger.info("Loading CIFAR-10 dataset via validated frozen-split pipeline...")
    _, _, (x_test, y_test) = load_cifar10_data(
        normalize=True,
        flatten_labels=True,
        splits_dir=SPLITS_DIR,
        seed=RANDOM_SEED,
    )

    # 4. Strict Test Set Isolation Verification
    assert len(x_test) == TEST_SAMPLE_COUNT, (
        f"Test sample count mismatch: expected {TEST_SAMPLE_COUNT}, got {len(x_test)}"
    )
    assert len(y_test) == TEST_SAMPLE_COUNT, (
        f"Test label count mismatch: expected {TEST_SAMPLE_COUNT}, got {len(y_test)}"
    )

    print(f"\n[Test Set Verification]")
    print(f"  • Total Test Samples : {len(x_test):,} images (Untouched during training)")
    print(f"  • Image Dimensions   : {x_test.shape[1:]} ({x_test.dtype})")
    print(f"  • Pixel Value Range  : [{x_test.min():.2f}, {x_test.max():.2f}] float32")
    print(f"  • Total Target Classes: {NUM_CLASSES} classes ({', '.join(CLASS_NAMES)})")

    # 5. Initialize Evaluator
    evaluator = BaselineEvaluator(
        checkpoint_path=checkpoint_filepath,
        class_names=CLASS_NAMES,
        metrics_dir=METRICS_DIR,
        figures_dir=FIGURES_DIR,
        logs_dir=LOGS_DIR,
    )

    # 6. Execute Evaluation
    metrics = evaluator.evaluate(
        x_test=x_test,
        y_test=y_test,
        batch_size=batch_size,
        save_artifacts=True,
        prefix="baseline",
    )

    # 7. Print Comprehensive Evaluation Summary
    print("\n" + "=" * 80)
    print("                    FINAL TEST EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  • Checkpoint Evaluated : {checkpoint_filepath.name}")
    print(f"  • Test Sample Count    : {metrics['total_samples']:,}")
    print(f"  • Number of Classes    : {metrics['num_classes']}")
    print(f"  • Test Cross-Entropy Loss: {metrics['test_loss']:.4f}")
    print(f"  • Test Overall Accuracy  : {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy_percent']:.2f}%)")
    print(f"  • Macro Avg F1-Score     : {metrics['macro_avg']['f1_score']:.4f}")
    print(f"  • Weighted Avg F1-Score  : {metrics['weighted_avg']['f1_score']:.4f}")
    print("-" * 80)
    print("                  PER-CLASS CLASSIFICATION BREAKDOWN")
    print(format_metrics_table(metrics))
    print("=" * 80)
    print("                    GENERATED EVALUATION ARTIFACTS")
    print("=" * 80)
    print(f"  - Metrics JSON          : {METRICS_DIR / 'baseline_test_metrics.json'}")
    print(f"  - Classification CSV    : {METRICS_DIR / 'baseline_classification_report.csv'}")
    print(f"  - Confusion Matrix CSV  : {METRICS_DIR / 'baseline_confusion_matrix.csv'}")
    print(f"  - Confusion Matrix Plot : {FIGURES_DIR / 'baseline_confusion_matrix.png'}")
    print(f"  - Normalized CM Plot    : {FIGURES_DIR / 'baseline_confusion_matrix_normalized.png'}")
    print(f"  - Per-Class Metrics Bar : {FIGURES_DIR / 'baseline_per_class_metrics.png'}")
    print(f"  - Training Curves Plot  : {FIGURES_DIR / 'baseline_training_curves.png'}")
    print("=" * 80 + "\n")

    return metrics


if __name__ == "__main__":
    run_baseline_evaluation()
