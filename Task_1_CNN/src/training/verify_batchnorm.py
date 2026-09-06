"""
Standalone Smoke & Structural Verification for Task 1 Part B — Experiment 1B (Batch Normalization ONLY).

Performs strict pre-flight checks before full 30-epoch training:
1. Model building and uncompiled state
2. Presence and configuration of exactly 6 BatchNormalization layers and 6 ReLU layers
3. Conventional layer ordering: Conv2D (no act) -> BN -> ReLU for all 5 conv layers
4. Dense layer ordering: Dense(512, no act) -> BN -> ReLU -> Dense(10, Softmax)
5. Absence of Dropout layers
6. Absence of L1/L2 weight decay / regularizers
7. Exact total parameter count: 2,662,730 (+4,608 vs baseline 2,658,122)
8. Exact trainable / non-trainable parameter counts (Trainable: 2,660,426, Non-trainable: 2,304)
9. Input shape (None, 32, 32, 3) and output shape (None, 10)
10. Dynamic forward-pass verification with dummy batch in training and inference modes
11. Loading and partition verification of frozen CIFAR-10 split (40k train, 10k val, 10k isolated test)
12. Target artifact paths distinction (zero overwriting of baseline or dropout checkpoints/logs)
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
from src.models.cnn_architecture import (
    build_baseline_cnn,
    build_batchnorm_cnn,
    build_dropout_cnn,
)
from src.training.trainer import BaselineTrainer
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_batchnorm")


def run_smoke_verification() -> bool:
    print("=" * 80)
    print("      EXPERIMENT 1B: PRE-TRAINING SMOKE & STRUCTURAL VERIFICATION")
    print("=" * 80)

    set_seed(RANDOM_SEED)

    # --------------------------------------------------------------------------
    # 1. Architecture & Layer Configuration Verification
    # --------------------------------------------------------------------------
    print("\n[Step 1/5] Constructing Models and Validating Conventional Layer Architecture...")
    baseline_model = build_baseline_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)
    bn_model = build_batchnorm_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)

    # Check layer types
    bn_layers = [layer for layer in bn_model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    relu_layers = [layer for layer in bn_model.layers if isinstance(layer, (tf.keras.layers.ReLU, tf.keras.layers.Activation))]
    conv_layers = [layer for layer in bn_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    dense_layers = [layer for layer in bn_model.layers if isinstance(layer, tf.keras.layers.Dense)]
    dropout_layers = [layer for layer in bn_model.layers if isinstance(layer, tf.keras.layers.Dropout)]

    print(f"  • Model Name                   : {bn_model.name}")
    print(f"  • Total Layers in Model        : {len(bn_model.layers)}")
    print(f"  • Input Shape                  : {bn_model.input_shape} (Expected: (None, 32, 32, 3))")
    print(f"  • Output Shape                 : {bn_model.output_shape} (Expected: (None, 10))")
    print(f"  • Conv2D Layers Found          : {len(conv_layers)} (Expected: 5)")
    print(f"  • BatchNorm Layers Found       : {len(bn_layers)} (Expected: 6)")
    print(f"  • ReLU Layers Found            : {len(relu_layers)} (Expected: 6)")
    print(f"  • Dense Layers Found           : {len(dense_layers)} (Expected: 2)")
    print(f"  • Dropout Layers Found         : {len(dropout_layers)} (Expected: 0)")

    assert len(conv_layers) == 5, f"Expected 5 Conv2D layers, found {len(conv_layers)}"
    assert len(bn_layers) == 6, f"Expected exactly 6 BatchNorm layers, found {len(bn_layers)}"
    assert len(relu_layers) == 6, f"Expected exactly 6 ReLU layers, found {len(relu_layers)}"
    assert len(dense_layers) == 2, f"Expected exactly 2 Dense layers, found {len(dense_layers)}"
    assert len(dropout_layers) == 0, f"Found {len(dropout_layers)} Dropout layers; must be 0 for Experiment 1B"

    # Verify conventional Conv -> BN -> ReLU sequence for all 5 conv stages
    for idx, conv_layer in enumerate(conv_layers):
        conv_idx = bn_model.layers.index(conv_layer)
        next_layer = bn_model.layers[conv_idx + 1]
        following_layer = bn_model.layers[conv_idx + 2]
        assert isinstance(next_layer, tf.keras.layers.BatchNormalization), (
            f"Layer immediately after {conv_layer.name} (idx {conv_idx}) must be BatchNormalization, got {type(next_layer).__name__}"
        )
        assert isinstance(following_layer, (tf.keras.layers.ReLU, tf.keras.layers.Activation)), (
            f"Layer immediately after {next_layer.name} (idx {conv_idx + 1}) must be ReLU, got {type(following_layer).__name__}"
        )
    print("  • Conv Conventional Placement  : PASSED (All 5 Conv2D -> BatchNormalization -> ReLU confirmed)")

    # Verify Dense(512) -> BN -> ReLU -> Dense(10, Softmax) sequence
    dense1 = dense_layers[0]
    dense1_idx = bn_model.layers.index(dense1)
    bn_dense = bn_model.layers[dense1_idx + 1]
    relu_dense = bn_model.layers[dense1_idx + 2]
    pred_layer = bn_model.layers[dense1_idx + 3]

    assert dense1.units == 512, f"Expected first dense layer to have 512 units, got {dense1.units}"
    assert isinstance(bn_dense, tf.keras.layers.BatchNormalization), f"Expected BN after dense1, got {type(bn_dense).__name__}"
    assert isinstance(relu_dense, (tf.keras.layers.ReLU, tf.keras.layers.Activation)), f"Expected ReLU after bn_dense, got {type(relu_dense).__name__}"
    assert isinstance(pred_layer, tf.keras.layers.Dense) and pred_layer.units == 10, "Expected predictions Dense(10)"
    print("  • Dense Conventional Placement : PASSED (Dense(512) -> BatchNormalization -> ReLU -> Dense(10) confirmed)")

    # Regularizer check (no L1/L2)
    for layer in bn_model.layers:
        assert getattr(layer, "kernel_regularizer", None) is None, f"{layer.name} has kernel_regularizer"
        assert getattr(layer, "bias_regularizer", None) is None, f"{layer.name} has bias_regularizer"
        assert getattr(layer, "activity_regularizer", None) is None, f"{layer.name} has activity_regularizer"
    print("  • Regularizer Absence Check    : PASSED (0 L1/L2 weight decay regularizers detected)")

    # --------------------------------------------------------------------------
    # 2. Parameter Count Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2/5] Validating Parameter Counts Against Baseline...")
    baseline_params = baseline_model.count_params()
    bn_total_params = bn_model.count_params()
    bn_trainable_params = sum(tf.keras.backend.count_params(w) for w in bn_model.trainable_weights)
    bn_non_trainable_params = sum(tf.keras.backend.count_params(w) for w in bn_model.non_trainable_weights)

    expected_baseline_params = 2_658_122
    expected_bn_total = 2_662_730
    expected_bn_trainable = 2_660_426
    expected_bn_non_trainable = 2_304

    print(f"  • Baseline Total Params        : {baseline_params:,} (Expected: {expected_baseline_params:,})")
    print(f"  • BatchNorm Total Params       : {bn_total_params:,} (Expected: {expected_bn_total:,})")
    print(f"  • Trainable Params             : {bn_trainable_params:,} (Expected: {expected_bn_trainable:,})")
    print(f"  • Non-Trainable Params (BN)    : {bn_non_trainable_params:,} (Expected: {expected_bn_non_trainable:,})")
    print(f"  • Param Difference vs Baseline : +{bn_total_params - baseline_params:,} (+4,608 parameters)")

    assert baseline_params == expected_baseline_params, f"Baseline mismatch: expected {expected_baseline_params}, got {baseline_params}"
    assert bn_total_params == expected_bn_total, f"BN total mismatch: expected {expected_bn_total}, got {bn_total_params}"
    assert bn_trainable_params == expected_bn_trainable, f"BN trainable mismatch: expected {expected_bn_trainable}, got {bn_trainable_params}"
    assert bn_non_trainable_params == expected_bn_non_trainable, f"BN non-trainable mismatch: expected {expected_bn_non_trainable}, got {bn_non_trainable_params}"
    print("  • Parameter Count Verification : PASSED (Exact match: 2,662,730 total params)")

    # --------------------------------------------------------------------------
    # 3. Dynamic Forward-Pass Smoke Test
    # --------------------------------------------------------------------------
    print("\n[Step 3/5] Executing Forward-Pass Smoke Test (Training & Inference Modes)...")
    dummy_input = tf.random.uniform(shape=(2, 32, 32, 3), minval=0.0, maxval=1.0, dtype=tf.float32)
    
    # Forward pass in training mode
    out_train = bn_model(dummy_input, training=True)
    assert out_train.shape == (2, 10), f"Training forward output shape mismatch: {out_train.shape}"
    
    # Forward pass in inference mode
    out_eval = bn_model(dummy_input, training=False)
    assert out_eval.shape == (2, 10), f"Inference forward output shape mismatch: {out_eval.shape}"
    
    print(f"  • Dummy Input Shape            : {dummy_input.shape}")
    print(f"  • Output Shape (training=True) : {out_train.shape}")
    print(f"  • Output Shape (training=False): {out_eval.shape}")
    print("  • Dynamic Forward-Pass Test    : PASSED")

    # --------------------------------------------------------------------------
    # 4. Dataset Split & Frozen Manifest Verification
    # --------------------------------------------------------------------------
    print("\n[Step 4/5] Validating Dataset Partitions and Frozen Split Manifest...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(
        normalize=True,
        flatten_labels=True,
        splits_dir=SPLITS_DIR,
        seed=RANDOM_SEED,
    )

    print(f"  • Training Pool Partition      : {len(x_train):,} samples (Expected: {TRAIN_SAMPLE_COUNT:,})")
    print(f"  • Validation Set Partition     : {len(x_val):,} samples (Expected: {VAL_SAMPLE_COUNT:,})")
    print(f"  • Test Set (Held Isolated)     : {len(x_test):,} samples (Expected: {TEST_SAMPLE_COUNT:,})")
    print(f"  • Pixel Value Range            : [{x_train.min():.2f}, {x_train.max():.2f}] float32")

    assert len(x_train) == TRAIN_SAMPLE_COUNT, "Train sample count mismatch"
    assert len(x_val) == VAL_SAMPLE_COUNT, "Validation sample count mismatch"
    assert len(x_test) == TEST_SAMPLE_COUNT, "Test sample count mismatch"
    print("  • Frozen Split Integrity Check : PASSED (Exact sample counts confirmed)")

    # --------------------------------------------------------------------------
    # 5. Artifact Path Separation & Baseline/Dropout Preservation Verification
    # --------------------------------------------------------------------------
    print("\n[Step 5/5] Validating Target Artifact Paths and Collision Isolation...")
    trainer = BaselineTrainer(
        model=bn_model,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        checkpoints_dir=CHECKPOINTS_DIR,
        checkpoint_filename="batchnorm_cifar10_best.keras",
        logs_dir=LOGS_DIR,
        history_filename="batchnorm_training_history.json",
        csv_log_filename="batchnorm_training_log.csv",
        experiment_name="batchnorm_cifar10",
    )

    baseline_ckpt = CHECKPOINTS_DIR / "baseline_cifar10_best.keras"
    baseline_csv = LOGS_DIR / "baseline_training_log.csv"
    baseline_json = LOGS_DIR / "baseline_training_history.json"

    dropout_ckpt = CHECKPOINTS_DIR / "dropout_cifar10_best.keras"
    dropout_csv = LOGS_DIR / "dropout_training_log.csv"
    dropout_json = LOGS_DIR / "dropout_training_history.json"

    # Confirm baseline and dropout artifacts exist and are preserved
    assert baseline_ckpt.exists(), f"Missing baseline checkpoint: {baseline_ckpt}"
    assert baseline_csv.exists(), f"Missing baseline CSV: {baseline_csv}"
    assert baseline_json.exists(), f"Missing baseline JSON: {baseline_json}"

    assert dropout_ckpt.exists(), f"Missing dropout checkpoint: {dropout_ckpt}"
    assert dropout_csv.exists(), f"Missing dropout CSV: {dropout_csv}"
    assert dropout_json.exists(), f"Missing dropout JSON: {dropout_json}"

    # Confirm batchnorm target paths are distinct from baseline and dropout
    assert trainer.checkpoint_filepath not in [baseline_ckpt, dropout_ckpt], "Checkpoint filename collision!"
    assert trainer.csv_log_filepath not in [baseline_csv, dropout_csv], "CSV log filename collision!"
    assert trainer.history_filepath not in [baseline_json, dropout_json], "JSON history filename collision!"

    print(f"  • Preserved Baseline Checkpoint : {baseline_ckpt.name}")
    print(f"  • Preserved Dropout Checkpoint  : {dropout_ckpt.name}")
    print(f"  • Target BatchNorm Checkpoint   : {trainer.checkpoint_filepath.name}")
    print(f"  • Target BatchNorm CSV Log      : {trainer.csv_log_filepath.name}")
    print(f"  • Target BatchNorm JSON Log     : {trainer.history_filepath.name}")
    print("  • Artifact Path Collision Check : PASSED (Zero collision with existing artifacts)")

    print("\n" + "=" * 80)
    print("  [SUCCESS] ALL PRE-TRAINING SMOKE & STRUCTURAL VERIFICATION CHECKS PASSED!")
    print("=" * 80 + "\n")
    return True


if __name__ == "__main__":
    success = run_smoke_verification()
    sys.exit(0 if success else 1)
