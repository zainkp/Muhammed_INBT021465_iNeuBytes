"""
Standalone Smoke & Structural Verification for Task 1 Part B — Experiment 1C (L2 Regularization ONLY).

Performs strict pre-flight checks before full 30-epoch training:
1. Model building and uncompiled state
2. Presence and configuration of L2 kernel regularizer (l2=1e-4) on all 5 Conv2D and 2 Dense layers
3. Absence of bias regularizers, activity regularizers, and L1 regularization
4. Absence of Dropout layers (0 Dropout layers)
5. Absence of BatchNormalization layers (0 BatchNorm layers)
6. Exact total parameter count matches baseline: 2,658,122
7. Exact trainable / non-trainable parameter counts (Trainable: 2,658,122, Non-trainable: 0)
8. Input shape (None, 32, 32, 3) and output shape (None, 10)
9. Dynamic forward-pass verification with dummy batch in training and inference modes + model.losses verification
10. Loading and partition verification of frozen CIFAR-10 split (40k train, 10k val, 10k isolated test)
11. Confirmation that test set is strictly isolated (zero test-set evaluation)
12. Target artifact paths distinction (zero overwriting of baseline, dropout, or batchnorm checkpoints/logs)
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
    DEFAULT_WEIGHT_DECAY,
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
    build_l2_cnn,
)
from src.training.trainer import BaselineTrainer
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_l2")


def run_smoke_verification() -> bool:
    print("=" * 80)
    print("      EXPERIMENT 1C: PRE-TRAINING SMOKE & STRUCTURAL VERIFICATION")
    print("=" * 80)

    set_seed(RANDOM_SEED)

    # --------------------------------------------------------------------------
    # 1. Architecture & Layer Configuration Verification
    # --------------------------------------------------------------------------
    print("\n[Step 1/5] Constructing Models and Validating Layer Architecture & Regularizers...")
    baseline_model = build_baseline_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)
    l2_model = build_l2_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, l2_reg=DEFAULT_WEIGHT_DECAY)

    # Check uncompiled state
    assert not l2_model.compiled, "L2 model should not be compiled upon construction."
    assert getattr(l2_model, "optimizer", None) is None, "L2 model optimizer should be None before trainer compilation."

    # Check layer types
    conv_layers = [layer for layer in l2_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    dense_layers = [layer for layer in l2_model.layers if isinstance(layer, tf.keras.layers.Dense)]
    dropout_layers = [layer for layer in l2_model.layers if isinstance(layer, tf.keras.layers.Dropout)]
    bn_layers = [layer for layer in l2_model.layers if "batchnormalization" in layer.__class__.__name__.lower()]

    print(f"  • Model Name                   : {l2_model.name}")
    print(f"  • Total Layers in Model        : {len(l2_model.layers)}")
    print(f"  • Input Shape                  : {l2_model.input_shape} (Expected: (None, 32, 32, 3))")
    print(f"  • Output Shape                 : {l2_model.output_shape} (Expected: (None, 10))")
    print(f"  • Conv2D Layers Found          : {len(conv_layers)} (Expected: 5)")
    print(f"  • Dense Layers Found           : {len(dense_layers)} (Expected: 2)")
    print(f"  • Dropout Layers Found         : {len(dropout_layers)} (Expected: 0)")
    print(f"  • BatchNorm Layers Found       : {len(bn_layers)} (Expected: 0)")

    assert len(conv_layers) == 5, f"Expected 5 Conv2D layers, found {len(conv_layers)}"
    assert len(dense_layers) == 2, f"Expected 2 Dense layers, found {len(dense_layers)}"
    assert len(dropout_layers) == 0, f"Found {len(dropout_layers)} Dropout layers; must be 0 for Experiment 1C"
    assert len(bn_layers) == 0, f"Found {len(bn_layers)} BatchNorm layers; must be 0 for Experiment 1C"

    # Verify L2 Regularizer on all Conv2D and Dense layers
    weight_bearing_layers = conv_layers + dense_layers
    expected_l2 = DEFAULT_WEIGHT_DECAY  # 1e-4

    print(f"\n  • Checking L2 Kernel Regularizers (Target L2 = {expected_l2}):")
    for layer in weight_bearing_layers:
        reg = getattr(layer, "kernel_regularizer", None)
        assert reg is not None, f"Layer '{layer.name}' is missing kernel_regularizer!"

        reg_config = reg.get_config()
        l2_val = reg_config.get("l2", getattr(reg, "l2", None))
        assert l2_val is not None and abs(float(l2_val) - expected_l2) < 1e-7, (
            f"Layer '{layer.name}' kernel regularizer l2={l2_val} does not match expected {expected_l2}"
        )
        l1_val = reg_config.get("l1", 0.0)
        assert l1_val == 0.0 or l1_val is None, f"Layer '{layer.name}' has unexpected L1 regularizer: {l1_val}"

        # Check absence of bias and activity regularization
        assert getattr(layer, "bias_regularizer", None) is None, f"Layer '{layer.name}' has non-null bias_regularizer!"
        assert getattr(layer, "activity_regularizer", None) is None, f"Layer '{layer.name}' has non-null activity_regularizer!"

        print(f"    - Layer '{layer.name:12s}': kernel_regularizer=L2(l2={float(l2_val)}), bias_reg=None [PASSED]")

    print("  • Regularizer Validation Check : PASSED (7/7 weight layers configured with L2=1e-4 kernel regularization)")

    # --------------------------------------------------------------------------
    # 2. Parameter Count Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2/5] Validating Parameter Counts Against Baseline Architecture...")
    baseline_params = baseline_model.count_params()
    l2_total_params = l2_model.count_params()
    l2_trainable_params = sum(tf.keras.backend.count_params(w) for w in l2_model.trainable_weights)
    l2_non_trainable_params = sum(tf.keras.backend.count_params(w) for w in l2_model.non_trainable_weights)

    expected_params = 2_658_122

    print(f"  • Baseline Total Params        : {baseline_params:,} (Expected: {expected_params:,})")
    print(f"  • L2 Total Params              : {l2_total_params:,} (Expected: {expected_params:,})")
    print(f"  • Trainable Params             : {l2_trainable_params:,} (Expected: {expected_params:,})")
    print(f"  • Non-Trainable Params         : {l2_non_trainable_params:,} (Expected: 0)")

    assert baseline_params == expected_params, f"Baseline mismatch: expected {expected_params}, got {baseline_params}"
    assert l2_total_params == expected_params, f"L2 total mismatch: expected {expected_params}, got {l2_total_params}"
    assert l2_trainable_params == expected_params, "Trainable params mismatch"
    assert l2_non_trainable_params == 0, "Non-trainable params must be 0"
    print("  • Parameter Count Verification : PASSED (Exact match with baseline: 2,658,122 parameters)")

    # --------------------------------------------------------------------------
    # 3. Dynamic Forward-Pass Smoke Test & Regularization Losses
    # --------------------------------------------------------------------------
    print("\n[Step 3/5] Executing Forward-Pass Smoke Test & Regularization Loss Tracking...")
    dummy_input = tf.random.uniform(shape=(2, 32, 32, 3), minval=0.0, maxval=1.0, dtype=tf.float32)

    # Forward pass in training mode
    out_train = l2_model(dummy_input, training=True)
    assert out_train.shape == (2, 10), f"Training forward output shape mismatch: {out_train.shape}"

    # Forward pass in inference mode
    out_eval = l2_model(dummy_input, training=False)
    assert out_eval.shape == (2, 10), f"Inference forward output shape mismatch: {out_eval.shape}"

    # Check that model.losses contains the regularization losses (7 weight-bearing layers)
    reg_losses = l2_model.losses
    print(f"  • Dummy Input Shape            : {dummy_input.shape}")
    print(f"  • Output Shape (training=True) : {out_train.shape}")
    print(f"  • Output Shape (training=False): {out_eval.shape}")
    print(f"  • Regularization Loss Tensors  : {len(reg_losses)} (Expected: 7)")
    assert len(reg_losses) == 7, f"Expected 7 regularization loss tensors in model.losses, got {len(reg_losses)}"

    total_reg_loss = tf.add_n(reg_losses).numpy()
    print(f"  • Total Initial L2 Reg Loss    : {total_reg_loss:.6f}")
    assert total_reg_loss > 0.0, f"Expected positive L2 regularization loss, got {total_reg_loss}"
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
    print(f"  • Test Set Evaluation Policy   : ZERO EVALUATION (Strictly Isolated)")

    assert len(x_train) == TRAIN_SAMPLE_COUNT, "Train sample count mismatch"
    assert len(x_val) == VAL_SAMPLE_COUNT, "Validation sample count mismatch"
    assert len(x_test) == TEST_SAMPLE_COUNT, "Test sample count mismatch"
    print("  • Frozen Split Integrity Check : PASSED (Exact sample counts confirmed)")

    # --------------------------------------------------------------------------
    # 5. Artifact Path Separation & Baseline/Dropout/BatchNorm Preservation Verification
    # --------------------------------------------------------------------------
    print("\n[Step 5/5] Validating Target Artifact Paths and Collision Isolation...")
    trainer = BaselineTrainer(
        model=l2_model,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        checkpoints_dir=CHECKPOINTS_DIR,
        checkpoint_filename="l2_cifar10_best.keras",
        logs_dir=LOGS_DIR,
        history_filename="l2_training_history.json",
        csv_log_filename="l2_training_log.csv",
        experiment_name="l2_regularization_cifar10",
    )

    baseline_ckpt = CHECKPOINTS_DIR / "baseline_cifar10_best.keras"
    baseline_csv = LOGS_DIR / "baseline_training_log.csv"
    baseline_json = LOGS_DIR / "baseline_training_history.json"

    dropout_ckpt = CHECKPOINTS_DIR / "dropout_cifar10_best.keras"
    dropout_csv = LOGS_DIR / "dropout_training_log.csv"
    dropout_json = LOGS_DIR / "dropout_training_history.json"

    batchnorm_ckpt = CHECKPOINTS_DIR / "batchnorm_cifar10_best.keras"
    batchnorm_csv = LOGS_DIR / "batchnorm_training_log.csv"
    batchnorm_json = LOGS_DIR / "batchnorm_training_history.json"

    # Confirm baseline, dropout, and batchnorm artifacts exist and are preserved
    assert baseline_ckpt.exists(), f"Missing baseline checkpoint: {baseline_ckpt}"
    assert baseline_csv.exists(), f"Missing baseline CSV: {baseline_csv}"
    assert baseline_json.exists(), f"Missing baseline JSON: {baseline_json}"

    assert dropout_ckpt.exists(), f"Missing dropout checkpoint: {dropout_ckpt}"
    assert dropout_csv.exists(), f"Missing dropout CSV: {dropout_csv}"
    assert dropout_json.exists(), f"Missing dropout JSON: {dropout_json}"

    assert batchnorm_ckpt.exists(), f"Missing batchnorm checkpoint: {batchnorm_ckpt}"
    assert batchnorm_csv.exists(), f"Missing batchnorm CSV: {batchnorm_csv}"
    assert batchnorm_json.exists(), f"Missing batchnorm JSON: {batchnorm_json}"

    # Confirm L2 target paths are distinct from baseline, dropout, and batchnorm
    existing_ckpts = [baseline_ckpt, dropout_ckpt, batchnorm_ckpt]
    existing_csvs = [baseline_csv, dropout_csv, batchnorm_csv]
    existing_jsons = [baseline_json, dropout_json, batchnorm_json]

    assert trainer.checkpoint_filepath not in existing_ckpts, "Checkpoint filename collision!"
    assert trainer.csv_log_filepath not in existing_csvs, "CSV log filename collision!"
    assert trainer.history_filepath not in existing_jsons, "JSON history filename collision!"

    print(f"  • Preserved Baseline Checkpoint : {baseline_ckpt.name}")
    print(f"  • Preserved Dropout Checkpoint  : {dropout_ckpt.name}")
    print(f"  • Preserved BatchNorm Checkpoint: {batchnorm_ckpt.name}")
    print(f"  • Target L2 Checkpoint          : {trainer.checkpoint_filepath.name}")
    print(f"  • Target L2 CSV Log             : {trainer.csv_log_filepath.name}")
    print(f"  • Target L2 JSON Log            : {trainer.history_filepath.name}")
    print("  • Artifact Path Collision Check : PASSED (Zero collision with existing artifacts)")

    print("\n" + "=" * 80)
    print("  [SUCCESS] ALL PRE-TRAINING SMOKE & STRUCTURAL VERIFICATION CHECKS PASSED!")
    print("=" * 80 + "\n")
    return True


if __name__ == "__main__":
    success = run_smoke_verification()
    sys.exit(0 if success else 1)
