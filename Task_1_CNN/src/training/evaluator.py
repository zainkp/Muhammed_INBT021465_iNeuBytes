"""
Evaluation Engine for Baseline CIFAR-10 Convolutional Neural Network.

This module implements the model evaluation engine for Task 1:
- Loads trained model checkpoints (.keras) safely.
- Performs inference and batch evaluation on the isolated 10,000-image CIFAR-10 test set.
- Computes overall test loss, test accuracy, sample counts, and class counts.
- Generates precision, recall, F1-scores (per-class, macro, weighted).
- Computes raw and row-normalized 10x10 confusion matrices.
- Saves structured evaluation artifacts into results/metrics/ and figures/.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# ------------------------------------------------------------------------------
# Config & Path Handling (Support both module and direct execution)
# ------------------------------------------------------------------------------
try:
    from src.config import (
        CHECKPOINTS_DIR,
        CLASS_NAMES,
        DEFAULT_BATCH_SIZE,
        FIGURES_DIR,
        LOGS_DIR,
        METRICS_DIR,
        NUM_CLASSES,
        TEST_SAMPLE_COUNT,
    )
    from src.utils.metrics import (
        compute_classification_metrics,
        format_metrics_table,
        save_classification_report_csv,
        save_confusion_matrix_csv,
        save_metrics_json,
    )
    from src.utils.plotting import (
        plot_confusion_matrix,
        plot_per_class_metrics,
        plot_training_history,
    )
except (ImportError, ModuleNotFoundError):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import (
        CHECKPOINTS_DIR,
        CLASS_NAMES,
        DEFAULT_BATCH_SIZE,
        FIGURES_DIR,
        LOGS_DIR,
        METRICS_DIR,
        NUM_CLASSES,
        TEST_SAMPLE_COUNT,
    )
    from src.utils.metrics import (
        compute_classification_metrics,
        format_metrics_table,
        save_classification_report_csv,
        save_confusion_matrix_csv,
        save_metrics_json,
    )
    from src.utils.plotting import (
        plot_confusion_matrix,
        plot_per_class_metrics,
        plot_training_history,
    )

# Setup module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


# ------------------------------------------------------------------------------
# Baseline Evaluator Class
# ------------------------------------------------------------------------------
class BaselineEvaluator:
    """
    Evaluator for the Baseline CIFAR-10 Convolutional Neural Network.

    Manages model checkpoint loading, test-set inference, metric computation,
    confusion matrix generation, and artifact persistence.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[tf.keras.Model] = None,
        class_names: List[str] = CLASS_NAMES,
        metrics_dir: Union[str, Path] = METRICS_DIR,
        figures_dir: Union[str, Path] = FIGURES_DIR,
        logs_dir: Union[str, Path] = LOGS_DIR,
    ) -> None:
        """
        Initialize the Baseline Evaluator.

        Args:
            checkpoint_path (Optional[Union[str, Path]]): Path to saved .keras checkpoint.
            model (Optional[tf.keras.Model]): Pre-loaded or instantiated Keras model.
            class_names (List[str]): List of class label strings. Defaults to CLASS_NAMES.
            metrics_dir (Union[str, Path]): Target directory for metrics files. Defaults to METRICS_DIR.
            figures_dir (Union[str, Path]): Target directory for figures/plots. Defaults to FIGURES_DIR.
            logs_dir (Union[str, Path]): Directory containing training logs. Defaults to LOGS_DIR.
        """
        if checkpoint_path is None and model is None:
            # Default to best baseline checkpoint
            self.checkpoint_path = Path(CHECKPOINTS_DIR) / "baseline_cifar10_best.keras"
        elif checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            self.checkpoint_path = None

        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.metrics_dir = Path(metrics_dir)
        self.figures_dir = Path(figures_dir)
        self.logs_dir = Path(logs_dir)

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, checkpoint_path: Optional[Union[str, Path]] = None) -> tf.keras.Model:
        """
        Load the trained Keras model checkpoint from disk.

        Args:
            checkpoint_path (Optional[Union[str, Path]]): Path to checkpoint file.

        Returns:
            tf.keras.Model: Loaded Keras model instance.
        """
        path = Path(checkpoint_path) if checkpoint_path is not None else self.checkpoint_path
        if path is None or not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")

        logger.info(f"Loading best baseline model checkpoint from: {path}")
        # Note: compile=False ensures we can re-evaluate with pure loss or standard evaluate
        try:
            model = tf.keras.models.load_model(str(path))
            logger.info("Loaded model successfully with pre-existing compilation.")
        except Exception as e:
            logger.warning(f"Standard loading with compilation encountered: {e}. Retrying with compile=False...")
            model = tf.keras.models.load_model(str(path), compile=False)

        # Ensure model is compiled for evaluate()
        if not model.compiled:
            logger.info("Compiling loaded model with SparseCategoricalCrossentropy and SparseCategoricalAccuracy...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )

        self.model = model
        self.checkpoint_path = path
        return model

    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = DEFAULT_BATCH_SIZE,
        save_artifacts: bool = True,
        prefix: str = "baseline",
    ) -> Dict[str, Any]:
        """
        Execute comprehensive test evaluation on the CIFAR-10 test set.

        Guarantees:
        - Evaluates strictly on the provided test set.
        - Computes test loss, test accuracy, sample count, class count.
        - Calculates precision, recall, F1 (macro, weighted, per-class).
        - Computes 10x10 confusion matrix (raw and normalized).
        - Persists outputs to results/metrics/ and figures/ if save_artifacts is True.

        Args:
            x_test (np.ndarray): Normalized test images, shape (10000, 32, 32, 3).
            y_test (np.ndarray): 1D integer test labels, shape (10000,).
            batch_size (int): Mini-batch size for prediction/evaluation. Defaults to DEFAULT_BATCH_SIZE (64).
            save_artifacts (bool): Whether to persist JSON/CSV metrics and PNG plots. Defaults to True.
            prefix (str): File naming prefix for saved artifacts. Defaults to "baseline".

        Returns:
            Dict[str, Any]: Complete metrics dictionary.
        """
        if self.model is None:
            self.load_model()

        y_test_flat = np.asarray(y_test).ravel()
        num_samples = len(x_test)

        if num_samples != len(y_test_flat):
            raise ValueError(f"Mismatch: {num_samples} test images but {len(y_test_flat)} labels.")

        logger.info("=" * 75)
        logger.info(f"Starting Test Set Evaluation on {num_samples:,} Samples ({self.num_classes} Classes)")
        logger.info(f"  - Checkpoint Source : {self.checkpoint_path}")
        logger.info(f"  - Batch Size        : {batch_size}")
        logger.info("=" * 75)

        # 1. Compute Test Loss and Standard Test Accuracy via model.evaluate()
        logger.info("Evaluating test loss and accuracy via model.evaluate()...")
        eval_results = self.model.evaluate(x_test, y_test_flat, batch_size=batch_size, verbose=1)
        if isinstance(eval_results, list):
            test_loss = float(eval_results[0])
            test_accuracy = float(eval_results[1])
        else:
            test_loss = float(eval_results)
            test_accuracy = 0.0

        # 2. Compute Softmax Probabilities and Predicted Classes via model.predict()
        logger.info("Generating predictions via model.predict()...")
        y_probs = self.model.predict(x_test, batch_size=batch_size, verbose=1)
        y_pred = np.argmax(y_probs, axis=1)

        # 3. Calculate Comprehensive Classification Metrics & Confusion Matrix
        logger.info("Computing precision, recall, f1-scores, and confusion matrix...")
        metrics_dict = compute_classification_metrics(
            y_true=y_test_flat,
            y_pred=y_pred,
            class_names=self.class_names,
            digits=4,
        )

        # Augment with test loss and metadata
        metrics_dict["test_loss"] = round(test_loss, 4)
        metrics_dict["checkpoint_source"] = str(self.checkpoint_path) if self.checkpoint_path else "in-memory"
        metrics_dict["batch_size"] = batch_size

        # Verify computed accuracy matches evaluate accuracy closely
        computed_acc = metrics_dict["overall_accuracy"]
        logger.info(
            f"Test Loss: {test_loss:.4f} | Test Accuracy (evaluate): {test_accuracy:.4f} | Test Accuracy (computed): {computed_acc:.4f}"
        )

        # 4. Save Artifacts (JSON, CSV, Figures)
        if save_artifacts:
            # JSON Metrics
            json_path = self.metrics_dir / f"{prefix}_test_metrics.json"
            save_metrics_json(metrics_dict, json_path)

            # CSV Classification Report
            csv_report_path = self.metrics_dir / f"{prefix}_classification_report.csv"
            save_classification_report_csv(metrics_dict, csv_report_path)

            # CSV Confusion Matrix
            csv_cm_path = self.metrics_dir / f"{prefix}_confusion_matrix.csv"
            save_confusion_matrix_csv(metrics_dict["confusion_matrix"], self.class_names, csv_cm_path)

            # Confusion Matrix Figures (Raw counts & Normalized)
            cm_plot_path = self.figures_dir / f"{prefix}_confusion_matrix.png"
            plot_confusion_matrix(
                cm=metrics_dict["confusion_matrix"],
                class_names=self.class_names,
                filepath=cm_plot_path,
                title="CIFAR-10 Baseline CNN - Confusion Matrix (Counts)",
                normalize=False,
                cmap="Blues",
            )

            cm_norm_plot_path = self.figures_dir / f"{prefix}_confusion_matrix_normalized.png"
            plot_confusion_matrix(
                cm=metrics_dict["confusion_matrix"],
                class_names=self.class_names,
                filepath=cm_norm_plot_path,
                title="CIFAR-10 Baseline CNN - Confusion Matrix",
                normalize=True,
                cmap="Blues",
            )

            # Per-Class Metrics Bar Chart
            per_class_plot_path = self.figures_dir / f"{prefix}_per_class_metrics.png"
            plot_per_class_metrics(
                metrics_dict=metrics_dict,
                filepath=per_class_plot_path,
                title="CIFAR-10 Baseline CNN - Per-Class Metrics (Precision, Recall, F1)",
            )

            # Training Curves (if history log exists)
            history_file = self.logs_dir / "baseline_training_history.json"
            if not history_file.exists():
                history_file = self.logs_dir / "baseline_training_log.csv"
            if history_file.exists():
                curves_plot_path = self.figures_dir / f"{prefix}_training_curves.png"
                plot_training_history(
                    history_source=history_file,
                    filepath=curves_plot_path,
                    title="CIFAR-10 Baseline CNN - 30-Epoch Training & Validation Convergence",
                )

            logger.info("Saved all evaluation metrics and figures successfully.")

        return metrics_dict


# ------------------------------------------------------------------------------
# High-Level Functional Helper
# ------------------------------------------------------------------------------
def evaluate_baseline_checkpoint(
    x_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: Optional[Union[str, Path]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_artifacts: bool = True,
    prefix: str = "baseline",
) -> Dict[str, Any]:
    """
    Functional entry point to evaluate a baseline checkpoint on the test set.

    Args:
        x_test (np.ndarray): Test images array.
        y_test (np.ndarray): Test labels array.
        checkpoint_path (Optional[Union[str, Path]]): Path to .keras checkpoint file.
        batch_size (int): Mini-batch size. Defaults to DEFAULT_BATCH_SIZE (64).
        save_artifacts (bool): Whether to persist JSON/CSV/plots. Defaults to True.
        prefix (str): Filename prefix for outputs. Defaults to "baseline".

    Returns:
        Dict[str, Any]: Detailed evaluation metrics dictionary.
    """
    evaluator = BaselineEvaluator(checkpoint_path=checkpoint_path)
    evaluator.load_model()
    return evaluator.evaluate(
        x_test=x_test,
        y_test=y_test,
        batch_size=batch_size,
        save_artifacts=save_artifacts,
        prefix=prefix,
    )
