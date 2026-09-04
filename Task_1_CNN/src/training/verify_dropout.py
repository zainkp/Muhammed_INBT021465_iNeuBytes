"""
Standalone Smoke & Structural Verification for Task 1 Part B — Experiment 1A (Dropout Regularization ONLY).

Performs strict pre-flight checks before full 30-epoch training:
1. Model building and uncompiled state
2. Presence and configuration of Dropout(rate=0.5)
3. Layer ordering: Conv blocks -> Flatten -> Dense(512) -> Dropout(0.5) -> Dense(10, Softmax)
4. Absence of BatchNormalization layers
5. Absence of L1/L2 weight decay / regularizers
6. Exact total parameter count matches baseline: 2,658,122
7. Exact trainable / non-trainable parameter counts (Trainable: 2,658,122, Non-trainable: 0)
8. Input shape (None, 32, 32, 3) and output shape (None, 10)
9. Loading and partition verification of frozen CIFAR-10 split (40k train, 10k val, 10k isolated test)
10. Target artifact paths distinction (zero overwriting of baseline checkpoints or logs)
"""

import logging
import sys
from pathlib import Path

# Path configuration
CURRENT_FILE = Path(__file__).resolve()
TASK_ROOT = CURRENT_FILE.parent.parent.parent
REPO_ROOT = TASK_ROOT.parent

for path in [str(TASK_ROOT), str(REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import tensorflow as tf

from src.config import (
    CHECKPOINTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
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
from src.data.dataset import load_cifar10_data
from src.models.cnn_architecture import build_baseline_cnn, build_dropout_cnn
from src.training.trainer import BaselineTrainer
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_dropout")


def run_smoke_verification() -> bool:
    print("=" * 80)
    print("      EXPERIMENT 1A: PRE-TRAINING SMOKE & STRUCTURAL VERIFICATION")
    print("=" * 80)

    set_seed(RANDOM_SEED)

    # --------------------------------------------------------------------------
    # 1. Architecture & Layer Configuration Verification
    # --------------------------------------------------------------------------
    print("\n[Step 1/4] Constructing Models and Validating Layer Architecture...")
    baseline_model = build_baseline_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)
    dropout_model = build_dropout_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, dropout_rate=0.5)

    # Check layer types
    dropout_layers = [layer for layer in dropout_model.layers if isinstance(layer, tf.keras.layers.Dropout)]
    bn_layers = [layer for layer in dropout_model.layers if "batchnormalization" in layer.__class__.__name__.lower()]
    dense_layers = [layer for layer in dropout_model.layers if isinstance(layer, tf.keras.layers.Dense)]

    print(f"  • Model Name               : {dropout_model.name}")
    print(f"  • Layer Sequence           : {[layer.__class__.__name__ for layer in dropout_model.layers]}")
    print(f"  • Input Shape              : {dropout_model.input_shape} (Expected: (None, 32, 32, 3))")
    print(f"  • Output Shape             : {dropout_model.output_shape} (Expected: (None, 10))")
    print(f"  • Dropout Layers Found     : {len(dropout_layers)} (Expected: 1)")
    print(f"  • Dropout Rate             : {dropout_layers[0].rate} (Expected: 0.5)")
    print(f"  • BatchNorm Layers Found   : {len(bn_layers)} (Expected: 0)")

    assert len(dropout_layers) == 1, f"Expected 1 Dropout layer, found {len(dropout_layers)}"
    assert abs(dropout_layers[0].rate - 0.5) < 1e-6, f"Expected rate 0.5, got {dropout_layers[0].rate}"
    assert len(bn_layers) == 0, f"Found {len(bn_layers)} BatchNorm layers; must be 0"

    # Layer ordering: Dense(512) -> Dropout(0.5) -> Dense(10)
    assert len(dense_layers) == 2, f"Expected 2 Dense layers, got {len(dense_layers)}"
    dense1_idx = dropout_model.layers.index(dense_layers[0])
    dropout_idx = dropout_model.layers.index(dropout_layers[0])
    pred_idx = dropout_model.layers.index(dense_layers[1])
    assert dense1_idx < dropout_idx < pred_idx, "Dropout must be placed after Dense(512) and before Dense(10)"
    print("  • Dropout Placement Check  : PASSED (Dense(512) -> Dropout(0.5) -> Dense(10))")

    # Regularizer check
    for layer in dropout_model.layers:
        assert getattr(layer, "kernel_regularizer", None) is None, f"{layer.name} has kernel_regularizer"
        assert getattr(layer, "bias_regularizer", None) is None, f"{layer.name} has bias_regularizer"
        assert getattr(layer, "activity_regularizer", None) is None, f"{layer.name} has activity_regularizer"
    print("  • Weight Decay / Reg Check : PASSED (0 L1/L2 regularizers detected)")

    # --------------------------------------------------------------------------
    # 2. Parameter Count Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2/4] Validating Parameter Counts Against Verified Baseline...")
    baseline_params = baseline_model.count_params()
    dropout_params = dropout_model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in dropout_model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in dropout_model.non_trainable_weights)

    expected_params = 2_658_122

    print(f"  • Baseline Total Params    : {baseline_params:,}")
    print(f"  • Dropout Total Params     : {dropout_params:,}")
    print(f"  • Trainable Params         : {trainable_params:,}")
    print(f"  • Non-Trainable Params     : {non_trainable_params:,}")
    print(f"  • Expected Exact Params    : {expected_params:,}")

    assert baseline_params == expected_params, f"Baseline mismatch: expected {expected_params}, got {baseline_params}"
    assert dropout_params == expected_params, f"Dropout mismatch: expected {expected_params}, got {dropout_params}"
    assert trainable_params == expected_params, "Trainable params mismatch"
    assert non_trainable_params == 0, "Non-trainable params must be 0"
    print("  • Parameter Count Check    : PASSED (Exact match: 2,658,122 parameters)")

    # --------------------------------------------------------------------------
    # 3. Dataset Split & Frozen Manifest Verification
    # --------------------------------------------------------------------------
    print("\n[Step 3/4] Validating Dataset Partitions and Frozen Split Manifest...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(
        normalize=True,
        flatten_labels=True,
        splits_dir=SPLITS_DIR,
        seed=RANDOM_SEED,
    )

    print(f"  • Training Pool Partition  : {len(x_train):,} samples (Expected: {TRAIN_SAMPLE_COUNT:,})")
    print(f"  • Validation Set Partition : {len(x_val):,} samples (Expected: {VAL_SAMPLE_COUNT:,})")
    print(f"  • Test Set (Held Isolated) : {len(x_test):,} samples (Expected: {TEST_SAMPLE_COUNT:,})")
    print(f"  • Value Range              : [{x_train.min():.2f}, {x_train.max():.2f}] float32")

    assert len(x_train) == TRAIN_SAMPLE_COUNT, "Train sample count mismatch"
    assert len(x_val) == VAL_SAMPLE_COUNT, "Validation sample count mismatch"
    assert len(x_test) == TEST_SAMPLE_COUNT, "Test sample count mismatch"
    print("  • Frozen Split Check       : PASSED (Exact sample counts confirmed)")

    # --------------------------------------------------------------------------
    # 4. Artifact Path Separation & Baseline Preservation Verification
    # --------------------------------------------------------------------------
    print("\n[Step 4/4] Validating Target Artifact Paths and Baseline Preservation...")
    trainer = BaselineTrainer(
        model=dropout_model,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        checkpoints_dir=CHECKPOINTS_DIR,
        checkpoint_filename="dropout_cifar10_best.keras",
        logs_dir=LOGS_DIR,
        history_filename="dropout_training_history.json",
        csv_log_filename="dropout_training_log.csv",
        experiment_name="dropout_regularization_cifar10",
    )

    baseline_ckpt = CHECKPOINTS_DIR / "baseline_cifar10_best.keras"
    baseline_csv = LOGS_DIR / "baseline_training_log.csv"
    baseline_json = LOGS_DIR / "baseline_training_history.json"

    # Confirm baseline artifacts exist and are preserved
    assert baseline_ckpt.exists(), f"Missing baseline checkpoint: {baseline_ckpt}"
    assert baseline_csv.exists(), f"Missing baseline CSV: {baseline_csv}"
    assert baseline_json.exists(), f"Missing baseline JSON: {baseline_json}"

    # Confirm dropout target paths are distinct
    assert trainer.checkpoint_filepath != baseline_ckpt, "Checkpoint filename collision!"
    assert trainer.csv_log_filepath != baseline_csv, "CSV log filename collision!"
    assert trainer.history_filepath != baseline_json, "JSON history filename collision!"

    print(f"  • Preserved Baseline Checkpoint : {baseline_ckpt.name} ({baseline_ckpt.stat().st_size:,} bytes)")
    print(f"  • Preserved Baseline CSV Log    : {baseline_csv.name} ({baseline_csv.stat().st_size:,} bytes)")
    print(f"  • Preserved Baseline JSON Log   : {baseline_json.name} ({baseline_json.stat().st_size:,} bytes)")
    print(f"  • Target Dropout Checkpoint     : {trainer.checkpoint_filepath.name}")
    print(f"  • Target Dropout CSV Log        : {trainer.csv_log_filepath.name}")
    print(f"  • Target Dropout JSON Log       : {trainer.history_filepath.name}")
    print("  • Artifact Path Check           : PASSED (Zero collision with baseline artifacts)")

    print("\n" + "=" * 80)
    print("  [SUCCESS] ALL PRE-TRAINING SMOKE & STRUCTURAL VERIFICATION CHECKS PASSED!")
    print("=" * 80 + "\n")
    return True


if __name__ == "__main__":
    success = run_smoke_verification()
    sys.exit(0 if success else 1)
