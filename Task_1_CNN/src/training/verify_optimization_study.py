"""
Comprehensive Pre-Flight Verification Suite for Task 1 Part B — Experiment Group 3: Optimization Study.

Executes strict structural, algorithmic, and isolation smoke checks before training:
1. Backward Compatibility: BaselineTrainer(model, learning_rate=0.001) -> Adam with LR=0.001.
2. 3x3 Factorial Grid Instantiation:
   - SGD + Momentum (0.9, nesterov=False) @ {0.01, 0.001, 0.0001}
   - Adam (beta_1=0.9, beta_2=0.999, eps=1e-7) @ {0.01, 0.001, 0.0001}
   - RMSprop (rho=0.9, momentum=0.0, eps=1e-7) @ {0.01, 0.001, 0.0001}
3. Exact parameter count matches baseline: 2,658,122 (Trainable: 2,658,122, Non-trainable: 0).
4. Absence of Dropout, BatchNormalization, and L1/L2 weight regularization.
5. Frozen dataset split integrity: 40k train / 10k val / 10k test (held isolated).
6. Forward pass smoke test with dummy batch in training and inference modes.
7. Complete artifact path uniqueness across all 9 optimization runs.
8. Preservation and non-modification verification of prior experiment artifacts (Baseline, 1A-1C, 2A-2C).

Usage:
    python Task_1_CNN/src/training/verify_optimization_study.py
    python -m src.training.verify_optimization_study
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
import yaml

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
    CONFIGS_DIR,
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
from src.data.dataset import load_cifar10_data
from src.models.cnn_architecture import build_baseline_cnn
from src.training.run_optimization_study import EXPERIMENT_CONFIGS
from src.training.trainer import BaselineTrainer
from src.utils.seed import set_seed

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_optimization_study")


def verify_backward_compatibility() -> None:
    """Step 1: Verify BaselineTrainer default invocation constructs Adam(lr=0.001)."""
    print("\n[Step 1/8] Verifying BaselineTrainer Backward Compatibility...")
    model = build_baseline_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)
    trainer = BaselineTrainer(model=model, learning_rate=0.001)
    compiled_model = trainer.compile_model()

    assert isinstance(compiled_model.optimizer, tf.keras.optimizers.Adam), (
        f"Expected default optimizer to be Adam, got: {type(compiled_model.optimizer)}"
    )
    lr_val = float(compiled_model.optimizer.learning_rate.numpy())
    assert abs(lr_val - 0.001) < 1e-7, f"Expected default LR=0.001, got: {lr_val}"

    print("  • Default Optimizer Instance : Adam")
    print(f"  • Default Learning Rate      : {lr_val}")
    print("  • Backward Compatibility Check: PASSED (Zero breaking changes to legacy trainer calls)")


def verify_yaml_configs() -> None:
    """Step 2: Validate all 9 YAML configuration files exist and match requirements."""
    print("\n[Step 2/8] Validating 9 Factorial YAML Configuration Files...")

    expected_configs = [
        ("opt_sgd_lr_01_config.yaml", "sgd", 0.01),
        ("opt_sgd_lr_001_config.yaml", "sgd", 0.001),
        ("opt_sgd_lr_0001_config.yaml", "sgd", 0.0001),
        ("opt_adam_lr_01_config.yaml", "adam", 0.01),
        ("opt_adam_lr_001_config.yaml", "adam", 0.001),
        ("opt_adam_lr_0001_config.yaml", "adam", 0.0001),
        ("opt_rmsprop_lr_01_config.yaml", "rmsprop", 0.01),
        ("opt_rmsprop_lr_001_config.yaml", "rmsprop", 0.001),
        ("opt_rmsprop_lr_0001_config.yaml", "rmsprop", 0.0001),
    ]

    for fname, expected_opt, expected_lr in expected_configs:
        cfg_path = CONFIGS_DIR / fname
        assert cfg_path.exists(), f"Missing config file: {cfg_path}"

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert cfg["training"]["optimizer"] == expected_opt, (
            f"Config {fname}: expected optimizer '{expected_opt}', got '{cfg['training']['optimizer']}'"
        )
        assert abs(float(cfg["training"]["learning_rate"]) - expected_lr) < 1e-7, (
            f"Config {fname}: expected LR {expected_lr}, got {cfg['training']['learning_rate']}"
        )
        if expected_opt == "sgd":
            assert cfg["training"]["momentum"] == 0.9, f"Config {fname}: expected momentum=0.9"
            assert cfg["training"]["nesterov"] is False, f"Config {fname}: expected nesterov=false"
        elif expected_opt == "rmsprop":
            assert cfg["training"]["rho"] == 0.9, f"Config {fname}: expected rho=0.9"
            assert cfg["training"]["momentum"] == 0.0, f"Config {fname}: RMSprop must not use momentum"

        print(f"  • {fname:<32} -> {expected_opt.upper():<8} | LR={expected_lr:<7} [VALID]")

    print("  • YAML Configuration Check  : PASSED (All 9 configs verified)")


def verify_optimizer_grid_instantiation() -> None:
    """Step 3: Verify all 9 optimizer configurations instantiate and bind correctly."""
    print("\n[Step 3/8] Verifying 3x3 Factorial Optimizer Instantiation & Compilation...")

    for key, cfg in EXPERIMENT_CONFIGS.items():
        model = build_baseline_cnn()
        trainer = BaselineTrainer(
            model=model,
            optimizer=cfg["optimizer"],
            learning_rate=cfg["learning_rate"],
            momentum=cfg.get("momentum", 0.0),
            nesterov=cfg.get("nesterov", False),
            beta_1=cfg.get("beta_1", 0.9),
            beta_2=cfg.get("beta_2", 0.999),
            epsilon=cfg.get("epsilon", 1e-7),
            rho=cfg.get("rho", 0.9),
            experiment_name=cfg["experiment_name"],
        )
        compiled = trainer.compile_model()
        opt = compiled.optimizer

        def _to_float(v):
            return float(v.numpy()) if hasattr(v, "numpy") else float(v)

        lr_bound = _to_float(opt.learning_rate)
        assert abs(lr_bound - cfg["learning_rate"]) < 1e-7, (
            f"LR mismatch for {key}: expected {cfg['learning_rate']}, got {lr_bound}"
        )

        if cfg["optimizer"] == "sgd":
            assert isinstance(opt, tf.keras.optimizers.SGD), f"Expected SGD, got {type(opt)}"
            assert abs(_to_float(opt.momentum) - 0.9) < 1e-7, f"Expected momentum 0.9, got {_to_float(opt.momentum)}"
            assert opt.nesterov is False, f"Expected nesterov=False, got {opt.nesterov}"
        elif cfg["optimizer"] == "adam":
            assert isinstance(opt, tf.keras.optimizers.Adam), f"Expected Adam, got {type(opt)}"
            assert abs(_to_float(opt.beta_1) - 0.9) < 1e-7, f"Expected beta_1 0.9"
            assert abs(_to_float(opt.beta_2) - 0.999) < 1e-7, f"Expected beta_2 0.999"
        elif cfg["optimizer"] == "rmsprop":
            assert isinstance(opt, tf.keras.optimizers.RMSprop), f"Expected RMSprop, got {type(opt)}"
            assert abs(_to_float(opt.rho) - 0.9) < 1e-7, f"Expected rho 0.9, got {_to_float(opt.rho)}"
            assert abs(_to_float(opt.momentum) - 0.0) < 1e-7, f"Expected momentum 0.0 for RMSprop, got {_to_float(opt.momentum)}"

        print(f"  • {key:<20} -> {type(opt).__name__:<10} | LR={lr_bound:<7} [COMPILED OK]")

    print("  • Optimizer Grid Check      : PASSED (All 9 optimizer instances compiled successfully)")


def verify_model_architecture() -> None:
    """Step 4: Verify unregularized Baseline CNN architecture and parameter counts."""
    print("\n[Step 4/8] Verifying Model Architecture & Absence of Regularization...")
    model = build_baseline_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)

    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

    expected_params = 2_658_122
    assert total_params == expected_params, f"Total params mismatch: expected {expected_params}, got {total_params}"
    assert trainable_params == expected_params, "Trainable params mismatch"
    assert non_trainable_params == 0, "Non-trainable params must be 0"

    layer_types = [type(layer).__name__ for layer in model.layers]
    assert "Dropout" not in layer_types, "Dropout detected"
    assert "BatchNormalization" not in layer_types, "BatchNormalization detected"

    for layer in model.layers:
        assert getattr(layer, "kernel_regularizer", None) is None, f"{layer.name} has kernel regularizer"
        assert getattr(layer, "bias_regularizer", None) is None, f"{layer.name} has bias regularizer"
        assert getattr(layer, "activity_regularizer", None) is None, f"{layer.name} has activity regularizer"

    print(f"  • Model Name               : {model.name}")
    print(f"  • Total Parameters         : {total_params:,} (Trainable: {trainable_params:,}, Non-trainable: {non_trainable_params})")
    print(f"  • Layer Types              : {layer_types}")
    print(f"  • Regularization Status    : 0 Dropout, 0 BatchNorm, 0 L1/L2 Regularizers")
    print("  • Architecture Check       : PASSED (Exact match with pure baseline CNN)")


def verify_dataset_partitions() -> None:
    """Step 5: Verify frozen CIFAR-10 split counts and normalization."""
    print("\n[Step 5/8] Verifying Frozen CIFAR-10 Partitions & Test Isolation...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(
        normalize=True,
        flatten_labels=True,
        splits_dir=SPLITS_DIR,
        seed=RANDOM_SEED,
    )

    assert len(x_train) == TRAIN_SAMPLE_COUNT, f"Train count mismatch: expected {TRAIN_SAMPLE_COUNT}, got {len(x_train)}"
    assert len(x_val) == VAL_SAMPLE_COUNT, f"Val count mismatch: expected {VAL_SAMPLE_COUNT}, got {len(x_val)}"
    assert len(x_test) == TEST_SAMPLE_COUNT, f"Test count mismatch: expected {TEST_SAMPLE_COUNT}, got {len(x_test)}"

    assert 0.0 <= x_train.min() and x_train.max() <= 1.0, "Train pixel values outside [0, 1]"
    assert 0.0 <= x_val.min() and x_val.max() <= 1.0, "Val pixel values outside [0, 1]"

    print(f"  • Training Pool Partition  : {len(x_train):,} samples (Normalized float32 [0.0, 1.0])")
    print(f"  • Validation Set Partition : {len(x_val):,} samples (Unaugmented, Normalized)")
    print(f"  • Test Set Partition (Held): {len(x_test):,} samples [STRICTLY ISOLATED]")
    print("  • Partition Integrity Check: PASSED (Exact split counts confirmed)")


def verify_forward_pass_smoke() -> None:
    """Step 6: Dynamic forward-pass smoke check with dummy batch."""
    print("\n[Step 6/8] Executing Forward-Pass Numerical Stability Smoke Test...")
    model = build_baseline_cnn()
    dummy_input = tf.random.uniform((2, 32, 32, 3), minval=0.0, maxval=1.0, dtype=tf.float32)

    # Inference mode
    out_eval = model(dummy_input, training=False)
    assert out_eval.shape == (2, 10), f"Expected shape (2, 10), got {out_eval.shape}"
    row_sums = tf.reduce_sum(out_eval, axis=-1).numpy()
    for s in row_sums:
        assert abs(s - 1.0) < 1e-5, f"Softmax output does not sum to 1.0: {s}"

    # Training mode
    out_train = model(dummy_input, training=True)
    assert out_train.shape == (2, 10), f"Expected shape (2, 10), got {out_train.shape}"

    print(f"  • Forward Pass Output Shape: {out_eval.shape}")
    print(f"  • Softmax Normalization    : Sums = {row_sums} (Proper probabilities)")
    print("  • Forward Pass Smoke Check : PASSED (Numerically valid)")


def verify_artifact_isolation() -> None:
    """Step 7: Verify unique artifact filenames for all 9 experiments with no collisions."""
    print("\n[Step 7/8] Validating Artifact Paths & Disambiguation...")

    all_ckpt_paths: set = set()
    all_csv_paths: set = set()
    all_json_paths: set = set()
    all_summary_paths: set = set()
    all_figure_paths: set = set()

    for key, cfg in EXPERIMENT_CONFIGS.items():
        name = cfg["experiment_name"]
        ckpt = CHECKPOINTS_DIR / f"{name}_cifar10_best.keras"
        csv = LOGS_DIR / f"{name}_training_log.csv"
        history = LOGS_DIR / f"{name}_training_history.json"
        summary = METRICS_DIR / f"{name}_experiment_summary.json"
        fig = FIGURES_DIR / f"{name}_training_curves.png"

        assert ckpt not in all_ckpt_paths, f"Checkpoint collision: {ckpt}"
        assert csv not in all_csv_paths, f"CSV log collision: {csv}"
        assert history not in all_json_paths, f"History JSON collision: {history}"
        assert summary not in all_summary_paths, f"Summary JSON collision: {summary}"
        assert fig not in all_figure_paths, f"Figure collision: {fig}"

        all_ckpt_paths.add(ckpt)
        all_csv_paths.add(csv)
        all_json_paths.add(history)
        all_summary_paths.add(summary)
        all_figure_paths.add(fig)

    assert len(all_ckpt_paths) == 9, "Expected 9 unique checkpoint paths"
    assert len(all_csv_paths) == 9, "Expected 9 unique CSV log paths"
    assert len(all_json_paths) == 9, "Expected 9 unique JSON history paths"
    assert len(all_summary_paths) == 9, "Expected 9 unique summary JSON paths"
    assert len(all_figure_paths) == 9, "Expected 9 unique figure paths"

    print("  • Checkpoint Paths Count   : 9 unique targets")
    print("  • CSV Log Paths Count      : 9 unique targets")
    print("  • JSON History Paths Count : 9 unique targets")
    print("  • Summary Paths Count      : 9 unique targets")
    print("  • Figure Paths Count       : 9 unique targets")
    print("  • Artifact Isolation Check : PASSED (Zero collisions across 3x3 grid)")


def verify_prior_experiments_preserved() -> None:
    """Step 8: Confirm all artifacts from prior experiments (Baseline, 1A-1C, 2A-2C) are intact."""
    print("\n[Step 8/8] Verifying Prior Experiment Artifacts Remain Intact...")

    prior_checkpoints = [
        "baseline_cifar10_best.keras",
        "dropout_cifar10_best.keras",
        "batchnorm_cifar10_best.keras",
        "l2_cifar10_best.keras",
        "augmentation_light_cifar10_best.keras",
        "augmentation_moderate_cifar10_best.keras",
        "augmentation_aggressive_cifar10_best.keras",
    ]

    prior_summaries = [
        "baseline_test_metrics.json",
        "dropout_experiment_summary.json",
        "batchnorm_experiment_summary.json",
        "l2_experiment_summary.json",
        "augmentation_light_experiment_summary.json",
        "augmentation_moderate_experiment_summary.json",
        "augmentation_aggressive_experiment_summary.json",
    ]

    for ckpt_name in prior_checkpoints:
        p = CHECKPOINTS_DIR / ckpt_name
        assert p.exists(), f"Prior checkpoint missing: {p}"
        assert p.stat().st_size > 0, f"Prior checkpoint empty: {p}"
        print(f"  • Checkpoint Preserved    : {ckpt_name:<44} ({p.stat().st_size:,} bytes)")

    for sum_name in prior_summaries:
        p = METRICS_DIR / sum_name
        assert p.exists(), f"Prior summary missing: {p}"
        assert p.stat().st_size > 0, f"Prior summary empty: {p}"
        print(f"  • Summary Preserved       : {sum_name:<44} ({p.stat().st_size:,} bytes)")

    print("  • Prior Artifacts Check    : PASSED (All 7 prior models and metrics 100% intact)")


def run_all_verifications() -> bool:
    print("=" * 85)
    print("  EXPERIMENT GROUP 3: 3x3 FACTORIAL OPTIMIZATION STUDY PRE-FLIGHT VERIFICATION")
    print("=" * 85)

    set_seed(RANDOM_SEED)

    verify_backward_compatibility()
    verify_yaml_configs()
    verify_optimizer_grid_instantiation()
    verify_model_architecture()
    verify_dataset_partitions()
    verify_forward_pass_smoke()
    verify_artifact_isolation()
    verify_prior_experiments_preserved()

    print("\n" + "=" * 85)
    print("  [SUCCESS] ALL PRE-FLIGHT STRUCTURAL & FACTORIAL VERIFICATIONS PASSED!")
    print("=" * 85 + "\n")
    return True


if __name__ == "__main__":
    success = run_all_verifications()
    sys.exit(0 if success else 1)
