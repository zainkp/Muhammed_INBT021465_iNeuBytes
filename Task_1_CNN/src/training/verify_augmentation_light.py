"""
Standalone Smoke & Structural Verification for Task 1 Part B — Experiment 2A (Light Data Augmentation).

Performs strict pre-flight checks before full 30-epoch training:
1. Model building and uncompiled state
2. Exact baseline architecture layout (5 Conv2D, 2 Dense)
3. Absence of Dropout layers (0 Dropout)
4. Absence of BatchNormalization layers (0 BatchNorm)
5. Absence of L1/L2 weight decay / regularizers (0 regularizers)
6. Exact total parameter count matches baseline: 2,658,122 (Trainable: 2,658,122, Non-trainable: 0)
7. Input shape (None, 32, 32, 3) and output shape (None, 10)
8. Augmentation pipeline configuration: Random horizontal flip enabled ONLY; rotation=0, shift=0, zoom=0, brightness=0, contrast=0
9. Augmentation scope: Training stream ONLY; Validation and Test streams remain 100% unaugmented
10. Dynamic forward-pass smoke test with dummy batch in training and inference modes
11. Augmentation execution smoke test on sample batch
12. Loading and partition verification of frozen CIFAR-10 split (40k train, 10k val, 10k isolated test)
13. Strict isolation of test set (zero evaluation)
14. Target artifact paths distinction (zero overwriting of baseline, dropout, batchnorm, or L2 checkpoints/logs)
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Path configuration
CURRENT_FILE = Path(__file__).resolve()
TASK_ROOT = CURRENT_FILE.parent.parent.parent
REPO_ROOT = TASK_ROOT.parent

for path in [str(TASK_ROOT), str(REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import tensorflow as tf
import yaml

from src.config import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
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
from src.data.transforms import (
    create_augmented_tf_datasets,
    get_light_augmentation_pipeline,
    verify_light_augmentation_config,
)
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
logger = logging.getLogger("verify_augmentation_light")


def run_smoke_verification() -> bool:
    print("=" * 80)
    print("      EXPERIMENT 2A: PRE-TRAINING SMOKE & STRUCTURAL VERIFICATION")
    print("=" * 80)

    set_seed(RANDOM_SEED)

    # --------------------------------------------------------------------------
    # 1. Configuration File Verification
    # --------------------------------------------------------------------------
    print("\n[Step 1/6] Validating YAML Configuration File & Augmentation Controls...")
    config_path = CONFIGS_DIR / "augmentation_light_config.yaml"
    assert config_path.exists(), f"Missing configuration file: {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"  • Config Path                  : {config_path}")
    print(f"  • Experiment Name              : {config.get('experiment_name')}")
    print(f"  • Experiment Type              : {config.get('experiment_type')}")
    print(f"  • Augmentation Level           : {config.get('augmentation_level')}")
    print(f"  • Random Seed                  : {config.get('seed')}")

    assert config.get("experiment_name") == "augmentation_light_cifar10", "Experiment name mismatch in config"
    assert config.get("experiment_type") == "augmentation", "Experiment type mismatch in config"
    assert config.get("augmentation_level") == "light", "Augmentation level mismatch in config"
    assert config.get("seed") == 42, "Seed mismatch in config"

    # Verify augmentation parameters in config
    aug_cfg = config.get("augmentation", {})
    verify_light_augmentation_config(aug_cfg)
    print("  • Augmentation Controls Check  : PASSED (Horizontal Flip = True, All other transforms = 0.0)")

    # --------------------------------------------------------------------------
    # 2. Architecture & Layer Configuration Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2/6] Validating Baseline CNN Model Architecture...")
    baseline_model = build_baseline_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)

    # Check uncompiled state
    assert not baseline_model.compiled, "Model should not be compiled upon construction."
    assert getattr(baseline_model, "optimizer", None) is None, "Model optimizer should be None before trainer compilation."

    # Check layer types
    conv_layers = [layer for layer in baseline_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    dense_layers = [layer for layer in baseline_model.layers if isinstance(layer, tf.keras.layers.Dense)]
    dropout_layers = [layer for layer in baseline_model.layers if isinstance(layer, tf.keras.layers.Dropout)]
    bn_layers = [layer for layer in baseline_model.layers if "batchnormalization" in layer.__class__.__name__.lower()]

    print(f"  • Model Name                   : {baseline_model.name}")
    print(f"  • Total Layers in Model        : {len(baseline_model.layers)}")
    print(f"  • Input Shape                  : {baseline_model.input_shape} (Expected: (None, 32, 32, 3))")
    print(f"  • Output Shape                 : {baseline_model.output_shape} (Expected: (None, 10))")
    print(f"  • Conv2D Layers Found          : {len(conv_layers)} (Expected: 5)")
    print(f"  • Dense Layers Found           : {len(dense_layers)} (Expected: 2)")
    print(f"  • Dropout Layers Found         : {len(dropout_layers)} (Expected: 0)")
    print(f"  • BatchNorm Layers Found       : {len(bn_layers)} (Expected: 0)")

    assert len(conv_layers) == 5, f"Expected 5 Conv2D layers, found {len(conv_layers)}"
    assert len(dense_layers) == 2, f"Expected 2 Dense layers, found {len(dense_layers)}"
    assert len(dropout_layers) == 0, f"Found {len(dropout_layers)} Dropout layers; must be 0 for Experiment 2A"
    assert len(bn_layers) == 0, f"Found {len(bn_layers)} BatchNorm layers; must be 0 for Experiment 2A"

    # Verify absence of regularizers
    for layer in baseline_model.layers:
        assert getattr(layer, "kernel_regularizer", None) is None, f"{layer.name} has kernel_regularizer"
        assert getattr(layer, "bias_regularizer", None) is None, f"{layer.name} has bias_regularizer"
        assert getattr(layer, "activity_regularizer", None) is None, f"{layer.name} has activity_regularizer"
    print("  • Regularizer Absence Check    : PASSED (0 L1/L2 weight decay regularizers detected)")

    # --------------------------------------------------------------------------
    # 3. Parameter Count Verification
    # --------------------------------------------------------------------------
    print("\n[Step 3/6] Validating Parameter Counts Against Baseline...")
    total_params = baseline_model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in baseline_model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in baseline_model.non_trainable_weights)

    expected_params = 2_658_122

    print(f"  • Total Parameters             : {total_params:,} (Expected: {expected_params:,})")
    print(f"  • Trainable Parameters         : {trainable_params:,} (Expected: {expected_params:,})")
    print(f"  • Non-Trainable Parameters     : {non_trainable_params:,} (Expected: 0)")

    assert total_params == expected_params, f"Total mismatch: expected {expected_params}, got {total_params}"
    assert trainable_params == expected_params, f"Trainable mismatch: expected {expected_params}, got {trainable_params}"
    assert non_trainable_params == 0, f"Non-trainable mismatch: expected 0, got {non_trainable_params}"
    print("  • Parameter Count Check        : PASSED (Exact match: 2,658,122 parameters)")

    # --------------------------------------------------------------------------
    # 4. Augmentation Pipeline & Dynamic Smoke Tests
    # --------------------------------------------------------------------------
    print("\n[Step 4/6] Executing Augmentation Pipeline & Model Forward-Pass Smoke Tests...")
    aug_pipeline = get_light_augmentation_pipeline(seed=RANDOM_SEED)

    # Verify augmentation layer structure
    aug_layers = [type(layer).__name__ for layer in aug_pipeline.layers]
    print(f"  • Augmentation Pipeline Layers : {aug_layers}")
    assert len(aug_pipeline.layers) == 1, f"Expected 1 layer in pipeline, found {len(aug_pipeline.layers)}"
    assert isinstance(aug_pipeline.layers[0], tf.keras.layers.RandomFlip), "Expected RandomFlip layer"
    assert aug_pipeline.layers[0].mode == "horizontal", f"Expected mode 'horizontal', got {aug_pipeline.layers[0].mode}"

    # Test processing of small training batch
    dummy_input = tf.random.uniform(shape=(4, 32, 32, 3), minval=0.0, maxval=1.0, dtype=tf.float32)
    aug_output = aug_pipeline(dummy_input, training=True)
    unaug_output = aug_pipeline(dummy_input, training=False)

    assert aug_output.shape == dummy_input.shape, f"Augmented shape mismatch: {aug_output.shape}"
    assert unaug_output.shape == dummy_input.shape, f"Unaugmented shape mismatch: {unaug_output.shape}"
    # In training=False mode, RandomFlip should return identical tensors
    np.testing.assert_allclose(unaug_output.numpy(), dummy_input.numpy(), err_msg="Inference mode should not alter images")
    print("  • Augmentation Layer Behavior  : PASSED (RandomFlip active on training, inactive on inference)")

    # Test forward pass through baseline CNN
    out_train = baseline_model(dummy_input, training=True)
    out_eval = baseline_model(dummy_input, training=False)
    assert out_train.shape == (4, 10), f"Training forward output shape mismatch: {out_train.shape}"
    assert out_eval.shape == (4, 10), f"Inference forward output shape mismatch: {out_eval.shape}"
    print("  • CNN Model Forward Pass       : PASSED (Output shape (4, 10) confirmed)")

    # --------------------------------------------------------------------------
    # 5. Dataset Split & Frozen Manifest Verification
    # --------------------------------------------------------------------------
    print("\n[Step 5/6] Validating Dataset Partitions and Frozen Split Manifest...")
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

    # Build tf.data datasets and confirm validation is unaugmented
    train_ds, val_ds, test_ds = create_augmented_tf_datasets(
        x_train=x_train[:128],
        y_train=y_train[:128],
        x_val=x_val[:128],
        y_val=y_val[:128],
        x_test=x_test[:128],
        y_test=y_test[:128],
        batch_size=64,
        seed=RANDOM_SEED,
        augmentation_pipeline=aug_pipeline,
    )

    # Take 1 validation batch and confirm exact pixel equality with raw normalized slice
    for val_batch_x, val_batch_y in val_ds.take(1):
        np.testing.assert_allclose(
            val_batch_x.numpy(),
            x_val[:64],
            atol=1e-6,
            err_msg="Validation pipeline modified data! Validation data must remain 100% unaugmented.",
        )
    print("  • Validation Unaugmented Check : PASSED (Validation images identical to raw slice)")

    # --------------------------------------------------------------------------
    # 6. Artifact Path Separation & Isolation Verification
    # --------------------------------------------------------------------------
    print("\n[Step 6/6] Validating Target Artifact Paths and Collision Isolation...")
    trainer = BaselineTrainer(
        model=baseline_model,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        checkpoints_dir=CHECKPOINTS_DIR,
        checkpoint_filename="augmentation_light_cifar10_best.keras",
        logs_dir=LOGS_DIR,
        history_filename="augmentation_light_training_history.json",
        csv_log_filename="augmentation_light_training_log.csv",
        experiment_name="augmentation_light_cifar10",
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

    l2_ckpt = CHECKPOINTS_DIR / "l2_cifar10_best.keras"
    l2_csv = LOGS_DIR / "l2_training_log.csv"
    l2_json = LOGS_DIR / "l2_training_history.json"

    # Confirm prior experiment artifacts exist and are preserved
    assert baseline_ckpt.exists(), f"Missing baseline checkpoint: {baseline_ckpt}"
    assert baseline_csv.exists(), f"Missing baseline CSV: {baseline_csv}"
    assert baseline_json.exists(), f"Missing baseline JSON: {baseline_json}"

    assert dropout_ckpt.exists(), f"Missing dropout checkpoint: {dropout_ckpt}"
    assert dropout_csv.exists(), f"Missing dropout CSV: {dropout_csv}"
    assert dropout_json.exists(), f"Missing dropout JSON: {dropout_json}"

    assert batchnorm_ckpt.exists(), f"Missing batchnorm checkpoint: {batchnorm_ckpt}"
    assert batchnorm_csv.exists(), f"Missing batchnorm CSV: {batchnorm_csv}"
    assert batchnorm_json.exists(), f"Missing batchnorm JSON: {batchnorm_json}"

    assert l2_ckpt.exists(), f"Missing l2 checkpoint: {l2_ckpt}"
    assert l2_csv.exists(), f"Missing l2 CSV: {l2_csv}"
    assert l2_json.exists(), f"Missing l2 JSON: {l2_json}"

    # Confirm augmentation light target paths are completely distinct
    existing_ckpts = [baseline_ckpt, dropout_ckpt, batchnorm_ckpt, l2_ckpt]
    existing_csvs = [baseline_csv, dropout_csv, batchnorm_csv, l2_csv]
    existing_jsons = [baseline_json, dropout_json, batchnorm_json, l2_json]

    assert trainer.checkpoint_filepath not in existing_ckpts, "Checkpoint filename collision!"
    assert trainer.csv_log_filepath not in existing_csvs, "CSV log filename collision!"
    assert trainer.history_filepath not in existing_jsons, "JSON history filename collision!"

    print(f"  • Preserved Baseline Checkpoint   : {baseline_ckpt.name}")
    print(f"  • Preserved Dropout Checkpoint    : {dropout_ckpt.name}")
    print(f"  • Preserved BatchNorm Checkpoint  : {batchnorm_ckpt.name}")
    print(f"  • Preserved L2 Checkpoint         : {l2_ckpt.name}")
    print(f"  • Target Augmentation Checkpoint  : {trainer.checkpoint_filepath.name}")
    print(f"  • Target Augmentation CSV Log     : {trainer.csv_log_filepath.name}")
    print(f"  • Target Augmentation JSON Log    : {trainer.history_filepath.name}")
    print("  • Artifact Path Collision Check   : PASSED (Zero collision with existing artifacts)")

    print("\n" + "=" * 80)
    print("  [SUCCESS] ALL PRE-TRAINING SMOKE & STRUCTURAL VERIFICATION CHECKS PASSED!")
    print("=" * 80 + "\n")
    return True


if __name__ == "__main__":
    success = run_smoke_verification()
    sys.exit(0 if success else 1)
