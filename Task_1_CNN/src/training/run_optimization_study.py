"""
Core Execution Engine for Task 1 Part B — Experiment Group 3: Optimization Study.

This module provides the parameterized experiment runner for the 3x3 Factorial Optimization Study
on the unregularized Baseline CNN architecture on CIFAR-10.

Evaluates:
- SGD + Momentum (0.9) x {0.01, 0.001, 0.0001}
- Adam                 x {0.01, 0.001, 0.0001}
- RMSprop (rho=0.9)    x {0.01, 0.001, 0.0001}

Strict Scientific Controls:
1. Pure Baseline CNN architecture via build_baseline_cnn() (2,658,122 parameters).
2. Zero Dropout, zero BatchNormalization, zero L1/L2 weight regularization.
3. Zero data augmentation (pure unaugmented CIFAR-10 normalized images).
4. Deterministic random seed (42) across Python, NumPy, and TensorFlow.
5. Frozen CIFAR-10 split: 40,000 train / 10,000 val / 10,000 test.
6. Test set strictly isolated (zero test-set evaluation).
7. Fixed 30-epoch budget, batch size = 64.
8. Loss: SparseCategoricalCrossentropy(from_logits=False), metric: accuracy.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

# ------------------------------------------------------------------------------
# Source Module Imports
# ------------------------------------------------------------------------------
from src.config import (
    CHECKPOINTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
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
from src.models.cnn_architecture import build_baseline_cnn
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
logger = logging.getLogger("optimization_study")

# ------------------------------------------------------------------------------
# Experiment Registry (3x3 Full Factorial Grid)
# ------------------------------------------------------------------------------
EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    # 1. SGD + Momentum
    "opt_sgd_lr_01": {
        "experiment_name": "opt_sgd_lr_01",
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "nesterov": False,
        "display_name": "SGD + Momentum (0.9), LR=0.01",
    },
    "opt_sgd_lr_001": {
        "experiment_name": "opt_sgd_lr_001",
        "optimizer": "sgd",
        "learning_rate": 0.001,
        "momentum": 0.9,
        "nesterov": False,
        "display_name": "SGD + Momentum (0.9), LR=0.001",
    },
    "opt_sgd_lr_0001": {
        "experiment_name": "opt_sgd_lr_0001",
        "optimizer": "sgd",
        "learning_rate": 0.0001,
        "momentum": 0.9,
        "nesterov": False,
        "display_name": "SGD + Momentum (0.9), LR=0.0001",
    },
    # 2. Adam
    "opt_adam_lr_01": {
        "experiment_name": "opt_adam_lr_01",
        "optimizer": "adam",
        "learning_rate": 0.01,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
        "display_name": "Adam, LR=0.01",
    },
    "opt_adam_lr_001": {
        "experiment_name": "opt_adam_lr_001",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
        "display_name": "Adam, LR=0.001",
    },
    "opt_adam_lr_0001": {
        "experiment_name": "opt_adam_lr_0001",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
        "display_name": "Adam, LR=0.0001",
    },
    # 3. RMSprop
    "opt_rmsprop_lr_01": {
        "experiment_name": "opt_rmsprop_lr_01",
        "optimizer": "rmsprop",
        "learning_rate": 0.01,
        "rho": 0.9,
        "momentum": 0.0,
        "epsilon": 1e-7,
        "display_name": "RMSprop (rho=0.9), LR=0.01",
    },
    "opt_rmsprop_lr_001": {
        "experiment_name": "opt_rmsprop_lr_001",
        "optimizer": "rmsprop",
        "learning_rate": 0.001,
        "rho": 0.9,
        "momentum": 0.0,
        "epsilon": 1e-7,
        "display_name": "RMSprop (rho=0.9), LR=0.001",
    },
    "opt_rmsprop_lr_0001": {
        "experiment_name": "opt_rmsprop_lr_0001",
        "optimizer": "rmsprop",
        "learning_rate": 0.0001,
        "rho": 0.9,
        "momentum": 0.0,
        "epsilon": 1e-7,
        "display_name": "RMSprop (rho=0.9), LR=0.0001",
    },
}


def verify_unregularized_baseline_architecture(model: tf.keras.Model) -> None:
    """
    Perform structural and guardrail verification on the Baseline CNN model.

    Checks:
    - Input shape: (None, 32, 32, 3)
    - Output shape: (None, 10)
    - Total parameters == 2,658,122
    - Trainable parameters == 2,658,122
    - Non-trainable parameters == 0
    - Zero Dropout layers
    - Zero BatchNormalization layers
    - Zero L1/L2 weight regularizers
    """
    logger.info("Verifying pure baseline architecture integrity for Optimization Study...")

    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

    expected_params = 2_658_122
    assert total_params == expected_params, (
        f"Parameter count mismatch: expected {expected_params:,}, got {total_params:,}"
    )
    assert non_trainable_params == 0, f"Expected 0 non-trainable parameters, got {non_trainable_params}"
    assert trainable_params == expected_params, (
        f"Expected {expected_params:,} trainable parameters, got {trainable_params:,}"
    )

    layer_types = [type(layer).__name__ for layer in model.layers]
    assert "Dropout" not in layer_types, "Dropout detected in pure baseline model."
    assert "BatchNormalization" not in layer_types, "BatchNormalization detected in pure baseline model."

    for layer in model.layers:
        assert getattr(layer, "kernel_regularizer", None) is None, f"{layer.name} has kernel_regularizer"
        assert getattr(layer, "bias_regularizer", None) is None, f"{layer.name} has bias_regularizer"
        assert getattr(layer, "activity_regularizer", None) is None, f"{layer.name} has activity_regularizer"

    logger.info("[SUCCESS] Architecture confirmed: Pure Baseline CNN (2,658,122 parameters, 0 regularizers).")


def run_single_optimization_experiment(
    experiment_key: str,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, Any]]:
    """
    Execute a single 30-epoch training run from the Optimization Study grid.

    Args:
        experiment_key (str): Key identifying the experiment configuration.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, Any]]:
            Trained model, training history, and recorded summary dictionary.
    """
    if experiment_key not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment key: '{experiment_key}'. "
            f"Valid options are: {list(EXPERIMENT_CONFIGS.keys())}"
        )

    cfg = EXPERIMENT_CONFIGS[experiment_key]
    experiment_name = cfg["experiment_name"]
    optimizer_type = cfg["optimizer"]
    learning_rate = cfg["learning_rate"]
    momentum = cfg.get("momentum", 0.0)
    nesterov = cfg.get("nesterov", False)
    beta_1 = cfg.get("beta_1", 0.9)
    beta_2 = cfg.get("beta_2", 0.999)
    epsilon = cfg.get("epsilon", 1e-7)
    rho = cfg.get("rho", 0.9)
    display_name = cfg["display_name"]

    checkpoint_filename = f"{experiment_name}_cifar10_best.keras"
    csv_log_filename = f"{experiment_name}_training_log.csv"
    history_filename = f"{experiment_name}_training_history.json"
    summary_filename = f"{experiment_name}_experiment_summary.json"
    curves_filename = f"{experiment_name}_training_curves.png"

    # Ensure output directories exist
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 85)
    print(f"  TASK 1 PART B: EXPERIMENT GROUP 3 — {display_name.upper()}")
    print("=" * 85)

    # 1. Deterministic Reproducibility
    logger.info(f"Setting global random seed: {RANDOM_SEED}")
    set_seed(RANDOM_SEED)

    # 2. Load dataset via frozen split manifest
    logger.info("Loading CIFAR-10 data via frozen split pipeline...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(
        normalize=True,
        flatten_labels=True,
        splits_dir=SPLITS_DIR,
        seed=RANDOM_SEED,
    )

    # Verify split partition integrity
    assert len(x_train) == TRAIN_SAMPLE_COUNT, f"Train count mismatch: expected {TRAIN_SAMPLE_COUNT}, got {len(x_train)}"
    assert len(x_val) == VAL_SAMPLE_COUNT, f"Val count mismatch: expected {VAL_SAMPLE_COUNT}, got {len(x_val)}"
    assert len(x_test) == TEST_SAMPLE_COUNT, f"Test count mismatch: expected {TEST_SAMPLE_COUNT}, got {len(x_test)}"

    print(f"\n[Dataset Partitions Loaded]")
    print(f"  • Training Pool    : {len(x_train):,} samples (Shape: {x_train.shape}, dtype: {x_train.dtype})")
    print(f"  • Validation Set   : {len(x_val):,} samples (Shape: {x_val.shape}, dtype: {x_val.dtype})")
    print(f"  • Test Set (Held)  : {len(x_test):,} samples [STRICTLY ISOLATED - ZERO EVALUATION]")
    print(f"  • Normalization    : Pixels in [{x_train.min():.2f}, {x_train.max():.2f}] float32")

    # 3. Construct tf.data input pipelines (train and validation only)
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

    # 4. Instantiate unregularized baseline CNN architecture
    logger.info("Constructing unregularized Baseline CNN architecture...")
    model = build_baseline_cnn(
        input_shape=IMAGE_SHAPE,
        num_classes=NUM_CLASSES,
        name=f"{experiment_name}_cnn",
    )

    # 5. Architectural Verification
    verify_unregularized_baseline_architecture(model)

    print(f"\n[Model Architecture]")
    print(f"  • Model Name       : {model.name}")
    print(f"  • Total Parameters : {model.count_params():,}")
    print(f"  • Trainable Params : {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    print(f"  • Regularization   : None (Pure Baseline)")

    # 6. Instantiate BaselineTrainer with specific optimizer configuration
    logger.info(f"Initializing Trainer for {experiment_name}...")
    trainer = BaselineTrainer(
        model=model,
        optimizer=optimizer_type,
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        rho=rho,
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
    print(f"  • Optimizer        : {display_name}")
    print(f"  • Learning Rate    : {learning_rate}")
    if optimizer_type == "sgd":
        print(f"  • Momentum         : {momentum} (nesterov={nesterov})")
    elif optimizer_type == "rmsprop":
        print(f"  • Rho (discount)   : {rho} (momentum={momentum})")
    print(f"  • Loss Function    : SparseCategoricalCrossentropy(from_logits=False)")
    print(f"  • Metric Tracked   : SparseCategoricalAccuracy ('accuracy')")
    print(f"  • Epoch Budget     : {DEFAULT_EPOCHS} Epochs (Fixed)")
    print(f"  • Batch Size       : {DEFAULT_BATCH_SIZE}")
    print(f"  • Best Checkpoint  : {trainer.checkpoint_filepath}")
    print(f"  • CSV Metrics Log  : {trainer.csv_log_filepath}")
    print(f"  • JSON History Log : {trainer.history_filepath}")

    # Confirm non-collision with baseline and other existing artifacts
    baseline_checkpoint = CHECKPOINTS_DIR / "baseline_cifar10_best.keras"
    baseline_csv = LOGS_DIR / "baseline_training_log.csv"
    baseline_json = LOGS_DIR / "baseline_training_history.json"
    assert trainer.checkpoint_filepath != baseline_checkpoint, "Checkpoint filename collision with baseline!"
    assert trainer.csv_log_filepath != baseline_csv, "CSV log filename collision with baseline!"
    assert trainer.history_filepath != baseline_json, "JSON history filename collision with baseline!"

    # 8. Execute 30-Epoch Training Run with Timing
    print("\n" + "=" * 85)
    print(f"              STARTING 30-EPOCH TRAINING: {display_name.upper()}")
    print("=" * 85)

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
        "experiment_group": "optimization_study",
        "optimizer": optimizer_type.upper(),
        "optimizer_full": display_name,
        "learning_rate": learning_rate,
        "momentum": momentum if optimizer_type == "sgd" else None,
        "rho": rho if optimizer_type == "rmsprop" else None,
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)),
        "non_trainable_parameters": int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)),
        "batch_size": DEFAULT_BATCH_SIZE,
        "loss_function": "SparseCategoricalCrossentropy",
        "test_evaluation_performed": False,
        "test_set_isolated": True,
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
        "summary_json_path": str(METRICS_DIR / summary_filename),
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
        title=f"CIFAR-10 CNN ({display_name}) - Training & Validation Convergence",
    )

    # 12. Print Summary Table
    print("\n" + "=" * 85)
    print(f"          EXPERIMENT SUMMARY: {display_name.upper()}")
    print("=" * 85)
    print(f"  • Experiment Name          : {experiment_name}")
    print(f"  • Optimizer / LR           : {display_name}")
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
    print("=" * 85 + "\n")

    return model, history, summary_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment from the Optimization Study grid.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="Experiment key to run (e.g. opt_sgd_lr_01, opt_adam_lr_001, opt_rmsprop_lr_0001).",
    )
    args = parser.parse_args()
    run_single_optimization_experiment(args.experiment)
