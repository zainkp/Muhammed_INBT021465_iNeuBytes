"""
Execution Entry Point for Experiment 1A: Regularization — Baseline + Dropout ONLY.

This script executes the complete Part B Experiment 1A training pipeline on CIFAR-10:
1. Sets deterministic random seeds (42) across Python, NumPy, and TensorFlow.
2. Loads CIFAR-10 dataset using the validated frozen split manifest (40k train / 10k val / 10k test).
3. Keeps the 10,000-sample test set strictly isolated (zero test-set evaluation).
4. Constructs optimized tf.data.Dataset training and validation input pipelines.
5. Instantiates the uncompiled Dropout CNN architecture (identical to baseline + Dropout(0.5)).
6. Validates architecture integrity (parameter count, layer sequence, absence of BatchNorm/L1/L2).
7. Compiles and executes the fixed 30-epoch training protocol.
8. Persists checkpoint (dropout_cifar10_best.keras) and logs (dropout_training_log.csv, dropout_training_history.json).
9. Generates convergence plots (figures/dropout_training_curves.png).
10. Saves structured metrics summary (results/metrics/dropout_experiment_summary.json) and prints summary table.

Usage:
    Direct execution:
        python Task_1_CNN/src/training/run_dropout.py

    Module execution (from Task_1_CNN directory):
        python -m src.training.run_dropout

    Module execution (from workspace root):
        python -m Task_1_CNN.src.training.run_dropout
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import tensorflow as tf

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
# Source Module Imports
# ------------------------------------------------------------------------------
from src.config import (
    CHECKPOINTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    FIGURES_DIR,
    IMAGE_SHAPE,
    LOGS_DIR,
    METRICS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
    SPLITS_DIR,
    TEST_SAMPLE_COUNT,
    TRAIN_SAMPLE_COUNT,
    VAL_SAMPLE_COUNT,
)
from src.data.dataset import (
    create_tf_datasets,
    load_cifar10_data,
)
from src.models.cnn_architecture import build_dropout_cnn
from src.training.trainer import BaselineTrainer
from src.utils.plotting import plot_training_history
from src.utils.seed import set_seed

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_dropout")


def verify_dropout_architecture(model: tf.keras.Model, expected_dropout_rate: float = 0.5) -> None:
    """
    Perform structural and guardrail verification on the Dropout CNN model.

    Checks:
    - Input shape: (None, 32, 32, 3)
    - Output shape: (None, 10)
    - Total parameters == 2,658,122
    - Non-trainable parameters == 0
    - Exactly one Dropout layer with rate == expected_dropout_rate
    - Dropout placed after Dense(512) and before Dense(10)
    - Zero BatchNormalization layers
    - Zero L1/L2 weight regularizers
    """
    logger.info("Performing structural verification on Dropout CNN model...")

    # 1. Parameter counts
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

    expected_total_params = 2_658_122
    assert total_params == expected_total_params, (
        f"Parameter count mismatch: expected {expected_total_params:,}, got {total_params:,}"
    )
    assert non_trainable_params == 0, (
        f"Expected 0 non-trainable parameters, got {non_trainable_params}"
    )
    assert trainable_params == expected_total_params, (
        f"Expected {expected_total_params:,} trainable parameters, got {trainable_params:,}"
    )

    # 2. Check layers
    layer_types = [type(layer).__name__ for layer in model.layers]
    logger.info(f"Layer sequence: {layer_types}")

    # Ensure no BatchNorm
    assert "BatchNormalization" not in layer_types, "BatchNormalization detected, which violates Experiment 1A constraints."

    # Verify Dropout layer
    dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
    assert len(dropout_layers) == 1, f"Expected exactly 1 Dropout layer, found {len(dropout_layers)}"
    assert abs(dropout_layers[0].rate - expected_dropout_rate) < 1e-6, (
        f"Expected dropout rate {expected_dropout_rate}, got {dropout_layers[0].rate}"
    )

    # Verify layer positioning: Flatten -> Dense(512) -> Dropout -> Dense(10)
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    assert len(dense_layers) == 2, f"Expected 2 Dense layers, got {len(dense_layers)}"
    assert dense_layers[0].units == 512, f"Expected first dense layer to have 512 units, got {dense_layers[0].units}"
    assert dense_layers[1].units == NUM_CLASSES, f"Expected output dense layer to have {NUM_CLASSES} units, got {dense_layers[1].units}"

    # Verify index ordering
    dense1_idx = model.layers.index(dense_layers[0])
    dropout_idx = model.layers.index(dropout_layers[0])
    predictions_idx = model.layers.index(dense_layers[1])
    assert dense1_idx < dropout_idx < predictions_idx, (
        f"Dropout layer ordering violation: dense1_idx={dense1_idx}, dropout_idx={dropout_idx}, predictions_idx={predictions_idx}"
    )

    # Verify absence of weight regularizers
    for layer in model.layers:
        if hasattr(layer, "kernel_regularizer") and layer.kernel_regularizer is not None:
            raise AssertionError(f"Layer {layer.name} has kernel_regularizer, violating experiment constraints.")
        if hasattr(layer, "bias_regularizer") and layer.bias_regularizer is not None:
            raise AssertionError(f"Layer {layer.name} has bias_regularizer, violating experiment constraints.")
        if hasattr(layer, "activity_regularizer") and layer.activity_regularizer is not None:
            raise AssertionError(f"Layer {layer.name} has activity_regularizer, violating experiment constraints.")

    logger.info("[SUCCESS] Architecture verified: Exactly matches baseline with Dropout(0.5) added.")


def run_dropout_experiment() -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, Any]]:
    """
    Execute the end-to-end Experiment 1A (Dropout Regularization ONLY) on CIFAR-10.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, Any]]:
            Trained model, training history, and recorded experiment summary dictionary.
    """
    print("\n" + "=" * 80)
    print("      TASK 1 PART B: EXPERIMENT 1A — BASELINE + DROPOUT (0.5) ONLY")
    print("=" * 80)

    experiment_name = "dropout_regularization_cifar10"
    regularization_type = "Dropout"
    dropout_rate = 0.5

    checkpoint_filename = "dropout_cifar10_best.keras"
    csv_log_filename = "dropout_training_log.csv"
    history_filename = "dropout_training_history.json"
    summary_filename = "dropout_experiment_summary.json"
    curves_filename = "dropout_training_curves.png"

    # Ensure output directories exist
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Deterministic Reproducibility
    logger.info(f"Setting global random seed: {RANDOM_SEED}")
    set_seed(RANDOM_SEED)

    # 2. Load dataset via validated frozen split manifest
    logger.info("Loading CIFAR-10 data via frozen split pipeline...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(
        normalize=True,
        flatten_labels=True,
        splits_dir=SPLITS_DIR,
        seed=RANDOM_SEED,
    )

    # Split integrity validation
    assert len(x_train) == TRAIN_SAMPLE_COUNT, f"Train count mismatch: expected {TRAIN_SAMPLE_COUNT}, got {len(x_train)}"
    assert len(x_val) == VAL_SAMPLE_COUNT, f"Val count mismatch: expected {VAL_SAMPLE_COUNT}, got {len(x_val)}"
    assert len(x_test) == TEST_SAMPLE_COUNT, f"Test count mismatch: expected {TEST_SAMPLE_COUNT}, got {len(x_test)}"

    print(f"\n[Dataset Partitions Loaded]")
    print(f"  • Training Pool    : {len(x_train):,} samples (Shape: {x_train.shape}, dtype: {x_train.dtype})")
    print(f"  • Validation Set   : {len(x_val):,} samples (Shape: {x_val.shape}, dtype: {x_val.dtype})")
    print(f"  • Test Set (Held)  : {len(x_test):,} samples [STRICTLY ISOLATED - ZERO EVALUATION]")
    print(f"  • Normalization    : Pixels in [{x_train.min():.2f}, {x_train.max():.2f}] float32")

    # 3. Build tf.data input pipelines (train and validation only)
    logger.info(f"Constructing tf.data input pipelines (batch_size={DEFAULT_BATCH_SIZE})...")
    train_ds, val_ds, _ = create_tf_datasets(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle_buffer=10000,
        seed=RANDOM_SEED,
    )

    # 4. Instantiate Dropout CNN Architecture
    logger.info(f"Constructing Dropout CNN architecture (dropout_rate={dropout_rate})...")
    model = build_dropout_cnn(
        input_shape=IMAGE_SHAPE,
        num_classes=NUM_CLASSES,
        dropout_rate=dropout_rate,
        name="dropout_cifar10_cnn",
    )

    # 5. Perform architectural verification
    verify_dropout_architecture(model, expected_dropout_rate=dropout_rate)

    print(f"\n[Dropout CNN Architecture]")
    print(f"  • Model Name       : {model.name}")
    print(f"  • Input Shape      : {model.input_shape}")
    print(f"  • Output Shape     : {model.output_shape}")
    print(f"  • Total Parameters : {model.count_params():,}")
    print(f"  • Trainable Params : {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    print(f"  • Regularization   : Dropout(rate={dropout_rate}) after Dense(512)")

    # 6. Instantiate Trainer
    logger.info("Initializing Trainer for Experiment 1A...")
    trainer = BaselineTrainer(
        model=model,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        checkpoints_dir=CHECKPOINTS_DIR,
        checkpoint_filename=checkpoint_filename,
        logs_dir=LOGS_DIR,
        history_filename=history_filename,
        csv_log_filename=csv_log_filename,
        monitor_metric="val_accuracy",
        monitor_mode="max",
        experiment_name=experiment_name,
    )

    # 7. Compile Model
    trainer.compile_model()

    print(f"\n[Training Configuration]")
    print(f"  • Experiment Name  : {experiment_name}")
    print(f"  • Optimizer        : Adam (learning_rate={DEFAULT_LEARNING_RATE})")
    print(f"  • Loss Function    : SparseCategoricalCrossentropy(from_logits=False)")
    print(f"  • Metric Tracked   : SparseCategoricalAccuracy ('accuracy')")
    print(f"  • Epoch Budget     : {DEFAULT_EPOCHS} Epochs (Fixed)")
    print(f"  • Batch Size       : {DEFAULT_BATCH_SIZE}")
    print(f"  • Best Checkpoint  : {trainer.checkpoint_filepath}")
    print(f"  • CSV Metrics Log  : {trainer.csv_log_filepath}")
    print(f"  • JSON History Log : {trainer.history_filepath}")

    # Confirm non-collision with baseline files
    baseline_checkpoint = CHECKPOINTS_DIR / "baseline_cifar10_best.keras"
    baseline_csv = LOGS_DIR / "baseline_training_log.csv"
    baseline_json = LOGS_DIR / "baseline_training_history.json"
    assert trainer.checkpoint_filepath != baseline_checkpoint, "Checkpoint filename collision with baseline!"
    assert trainer.csv_log_filepath != baseline_csv, "CSV log filename collision with baseline!"
    assert trainer.history_filepath != baseline_json, "JSON history filename collision with baseline!"

    # 8. Execute 30-Epoch Training Run with Timing
    print("\n" + "=" * 80)
    print("              STARTING 30-EPOCH DROPOUT EXPERIMENT TRAINING")
    print("=" * 80)

    start_time = time.perf_counter()

    history = trainer.train(
        train_data=train_ds,
        val_data=val_ds,
        verbose=1,
    )

    total_training_time_sec = time.perf_counter() - start_time
    minutes = int(total_training_time_sec // 60)
    seconds = total_training_time_sec % 60
    formatted_time = f"{minutes}m {seconds:.2f}s"

    # 9. Compute Summary Metrics
    epochs_completed = len(history.epoch)
    final_train_loss = float(history.history["loss"][-1])
    final_train_acc = float(history.history["accuracy"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])

    best_val_acc = float(max(history.history["val_accuracy"]))
    best_val_epoch = int(history.history["val_accuracy"].index(best_val_acc) + 1)
    train_val_gap = float(final_train_acc - final_val_acc)

    summary_dict: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "regularization_type": regularization_type,
        "dropout_rate": dropout_rate,
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)),
        "non_trainable_parameters": int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)),
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "optimizer": "Adam",
        "loss_function": "SparseCategoricalCrossentropy",
        "training_time_seconds": round(total_training_time_sec, 2),
        "training_time_formatted": formatted_time,
        "epochs_completed": epochs_completed,
        "final_train_accuracy": round(final_train_acc, 4),
        "final_train_accuracy_percent": round(final_train_acc * 100.0, 2),
        "final_val_accuracy": round(final_val_acc, 4),
        "final_val_accuracy_percent": round(final_val_acc * 100.0, 2),
        "best_val_accuracy": round(best_val_acc, 4),
        "best_val_accuracy_percent": round(best_val_acc * 100.0, 2),
        "best_val_epoch": best_val_epoch,
        "final_train_loss": round(final_train_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "train_val_accuracy_gap": round(train_val_gap, 4),
        "train_val_accuracy_gap_percent": round(train_val_gap * 100.0, 2),
        "checkpoint_path": str(trainer.checkpoint_filepath),
        "csv_log_path": str(trainer.csv_log_filepath),
        "history_json_path": str(trainer.history_filepath),
        "training_curves_path": str(FIGURES_DIR / curves_filename),
    }

    # 10. Save Metrics Summary JSON
    summary_filepath = METRICS_DIR / summary_filename
    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Saved experiment summary JSON to: {summary_filepath}")

    # 11. Generate and Save Training Curves Plot
    curves_filepath = FIGURES_DIR / curves_filename
    logger.info(f"Generating training convergence curves at: {curves_filepath}...")
    plot_training_history(
        history_source=history.history,
        filepath=curves_filepath,
        title="CIFAR-10 CNN with Dropout (0.5) - Training & Validation Convergence",
    )

    # 12. Print Concise Experiment Summary Table
    print("\n" + "=" * 80)
    print("          EXPERIMENT 1A (DROPOUT REGULARIZATION) SUMMARY")
    print("=" * 80)
    print(f"  • Experiment Name          : {experiment_name}")
    print(f"  • Regularization Type      : {regularization_type} (rate={dropout_rate})")
    print(f"  • Dropout Rate             : {dropout_rate}")
    print(f"  • Total Parameters         : {model.count_params():,} (Trainable: {summary_dict['trainable_parameters']:,})")
    print(f"  • Training Time            : {formatted_time} ({total_training_time_sec:.2f} s)")
    print(f"  • Epochs Completed         : {epochs_completed} / {DEFAULT_EPOCHS}")
    print(f"  • Final Training Accuracy  : {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
    print(f"  • Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")
    print(f"  • Best Validation Accuracy : {best_val_acc:.4f} ({best_val_acc * 100:.2f}%) [Epoch {best_val_epoch}]")
    print(f"  • Final Training Loss      : {final_train_loss:.4f}")
    print(f"  • Final Validation Loss    : {final_val_loss:.4f}")
    print(f"  • Train-Validation Gap     : {train_val_gap:.4f} ({train_val_gap * 100:.2f} percentage points)")
    print(f"  • Saved Best Checkpoint    : {trainer.checkpoint_filepath}")
    print(f"  • Saved CSV Log            : {trainer.csv_log_filepath}")
    print(f"  • Saved History JSON       : {trainer.history_filepath}")
    print(f"  • Saved Summary JSON       : {summary_filepath}")
    print(f"  • Saved Curves Figure      : {curves_filepath}")
    print("=" * 80 + "\n")

    return model, history, summary_dict


if __name__ == "__main__":
    run_dropout_experiment()
