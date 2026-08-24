"""
Execution Entry Point for Baseline CNN Training on CIFAR-10.

This script executes the complete baseline training pipeline for Task 1:
1. Sets deterministic random seeds across Python, NumPy, and TensorFlow.
2. Loads CIFAR-10 dataset using the validated frozen split manifest (40k train / 10k val / 10k test).
3. Keeps the 10,000-sample test set strictly isolated for downstream evaluation.
4. Constructs optimized tf.data.Dataset training and validation input pipelines.
5. Instantiates the uncompiled baseline CNN architecture.
6. Instantiates and compiles the BaselineTrainer with Adam optimizer and Sparse Categorical Crossentropy.
7. Executes the fixed 30-epoch training protocol.
8. Persists the best validation checkpoint (.keras) and training logs (CSV and JSON) to permanent project directories.

Usage:
    Direct execution:
        python Task_1_CNN/src/training/run_baseline.py

    Module execution (from Task_1_CNN directory):
        python -m src.training.run_baseline

    Module execution (from workspace root):
        python -m Task_1_CNN.src.training.run_baseline
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

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
# Source Module Imports (Reuse existing codebase components without duplication)
# ------------------------------------------------------------------------------
from src.config import (
    CHECKPOINTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    IMAGE_SHAPE,
    LOGS_DIR,
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
from src.utils.seed import set_seed

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_baseline")


def run_baseline_training() -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Execute the end-to-end baseline training run on CIFAR-10.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]:
            The trained Keras model instance and the recorded training history.
    """
    print("\n" + "=" * 80)
    print("      CIFAR-10 BASELINE CNN: 30-EPOCH TRAINING EXECUTION")
    print("=" * 80)

    # 1. Ensure deterministic reproducibility
    logger.info(f"Setting global random seed: {RANDOM_SEED}")
    set_seed(RANDOM_SEED)

    # 2. Load dataset and verify frozen split
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
    print(f"  • Test Set (Held)  : {len(x_test):,} samples [ISOLATED - NOT USED DURING TRAINING]")
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

    # 4. Instantiate Baseline CNN Architecture
    logger.info("Constructing Baseline CNN architecture...")
    model = build_baseline_cnn(
        input_shape=IMAGE_SHAPE,
        num_classes=NUM_CLASSES,
        name="baseline_cifar10_cnn",
    )

    print(f"\n[Baseline CNN Architecture]")
    print(f"  • Model Name       : {model.name}")
    print(f"  • Input Shape      : {model.input_shape}")
    print(f"  • Output Shape     : {model.output_shape}")
    print(f"  • Total Parameters : {model.count_params():,}")
    print(f"  • Trainable Params : {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")

    # 5. Instantiate BaselineTrainer
    logger.info("Initializing BaselineTrainer...")
    trainer = BaselineTrainer(
        model=model,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        checkpoints_dir=CHECKPOINTS_DIR,
        checkpoint_filename="baseline_cifar10_best.keras",
        logs_dir=LOGS_DIR,
        history_filename="baseline_training_history.json",
        csv_log_filename="baseline_training_log.csv",
        monitor_metric="val_accuracy",
        monitor_mode="max",
    )

    # 6. Compile Model
    trainer.compile_model()

    print(f"\n[Training Configuration]")
    print(f"  • Optimizer        : Adam (learning_rate={DEFAULT_LEARNING_RATE})")
    print(f"  • Loss Function    : SparseCategoricalCrossentropy(from_logits=False)")
    print(f"  • Metric Tracked   : SparseCategoricalAccuracy ('accuracy')")
    print(f"  • Epoch Budget     : {DEFAULT_EPOCHS} Epochs (Fixed)")
    print(f"  • Batch Size       : {DEFAULT_BATCH_SIZE}")
    print(f"  • Best Checkpoint  : {trainer.checkpoint_filepath}")
    print(f"  • CSV Metrics Log  : {trainer.csv_log_filepath}")
    print(f"  • JSON History Log : {trainer.history_filepath}")

    # 7. Execute 30-Epoch Baseline Training
    print("\n" + "=" * 80)
    print("              STARTING 30-EPOCH BASELINE MODEL TRAINING")
    print("=" * 80)

    history = trainer.train(
        train_data=train_ds,
        val_data=val_ds,
        verbose=1,
    )

    # 8. Post-Training Summary
    best_val_acc = max(history.history["val_accuracy"])
    best_val_epoch = history.history["val_accuracy"].index(best_val_acc) + 1
    final_train_acc = history.history["accuracy"][-1]
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    print("\n" + "=" * 80)
    print("                  BASELINE TRAINING RUN COMPLETE")
    print("=" * 80)
    print(f"  • Total Epochs Run       : {len(history.epoch)}")
    print(f"  • Final Training Loss    : {final_train_loss:.4f}")
    print(f"  • Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
    print(f"  • Final Validation Loss  : {final_val_loss:.4f}")
    print(f"  • Best Validation Acc    : {best_val_acc:.4f} ({best_val_acc * 100:.2f}%) [Epoch {best_val_epoch}]")
    print(f"  • Saved Best Checkpoint  : {trainer.checkpoint_filepath}")
    print(f"  • Saved CSV Log          : {trainer.csv_log_filepath}")
    print(f"  • Saved History JSON     : {trainer.history_filepath}")
    print("=" * 80 + "\n")

    return model, history


if __name__ == "__main__":
    run_baseline_training()
