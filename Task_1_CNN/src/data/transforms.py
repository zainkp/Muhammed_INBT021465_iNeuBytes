"""
Data Preprocessing and Augmentation Pipelines for 32x32 CIFAR-10 Images.

This module provides data transformation and augmentation pipelines using Keras
preprocessing layers (tf.keras.layers) and TensorFlow tf.data routines.

Experiment 2A (Light Augmentation) Specifications:
- Random horizontal flip ONLY: tf.keras.layers.RandomFlip(mode="horizontal")
- Rotation: 0.0 (strictly disabled)
- Translation / Shift: 0.0 (strictly disabled)
- Zoom: 0.0 (strictly disabled)
- Brightness / Contrast adjustments: 0.0 (strictly disabled)
- Cropping / Resizing: None (32x32 image dimensions preserved)
- Augmentation is applied exclusively to the training dataset stream.
- Validation and test datasets remain 100% unaugmented.
- Source CIFAR-10 data on disk and in memory is not permanently modified.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ------------------------------------------------------------------------------
# Config & Path Handling (Support both module and direct execution)
# ------------------------------------------------------------------------------
try:
    from src.config import DEFAULT_BATCH_SIZE, IMAGE_SHAPE, RANDOM_SEED
except (ImportError, ModuleNotFoundError):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import DEFAULT_BATCH_SIZE, IMAGE_SHAPE, RANDOM_SEED

# Setup module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


# ------------------------------------------------------------------------------
# Light Augmentation Layer Builder
# ------------------------------------------------------------------------------
def get_light_augmentation_pipeline(
    seed: int = RANDOM_SEED,
    name: str = "light_augmentation_pipeline",
) -> tf.keras.Sequential:
    """
    Construct and return a Keras Sequential preprocessing pipeline for Light Data Augmentation.

    Strict Experimental Controls (Experiment 2A):
    - Random horizontal flip ONLY: tf.keras.layers.RandomFlip(mode="horizontal")
    - Rotation: 0.0 (None)
    - Shift/Translation: 0.0 (None)
    - Zoom: 0.0 (None)
    - Brightness/Contrast: 0.0 (None)
    - No resizing or cropping.

    Args:
        seed (int): Deterministic random seed for Keras augmentation layer. Defaults to RANDOM_SEED (42).
        name (str): Layer name for the Sequential pipeline. Defaults to "light_augmentation_pipeline".

    Returns:
        tf.keras.Sequential: A Sequential model containing exactly the RandomFlip layer.
    """
    augmentation_model = tf.keras.Sequential(
        [
            layers.Input(shape=IMAGE_SHAPE, name="augmentation_input"),
            layers.RandomFlip(mode="horizontal", seed=seed, name="random_horizontal_flip"),
        ],
        name=name,
    )
    logger.info(
        f"Built light augmentation pipeline '{name}' with RandomFlip(mode='horizontal', seed={seed}). "
        f"All other augmentations (rotation, shift, zoom, brightness, contrast) are strictly disabled."
    )
    return augmentation_model


# ------------------------------------------------------------------------------
# Configurable Augmentation Pipeline Builder
# ------------------------------------------------------------------------------
def build_augmentation_pipeline(
    horizontal_flip: bool = True,
    rotation_factor: float = 0.0,
    width_shift_factor: float = 0.0,
    height_shift_factor: float = 0.0,
    zoom_factor: float = 0.0,
    brightness_factor: float = 0.0,
    contrast_factor: float = 0.0,
    seed: int = RANDOM_SEED,
    name: str = "augmentation_pipeline",
) -> tf.keras.Sequential:
    """
    Build a configurable data augmentation Sequential model.

    For Light Augmentation (Experiment 2A), only horizontal_flip is True and
    all numeric factors must be 0.0.

    Args:
        horizontal_flip (bool): Whether to enable random horizontal flipping.
        rotation_factor (float): Float representing rotation range as a fraction of 2pi.
        width_shift_factor (float): Float representing horizontal shift range.
        height_shift_factor (float): Float representing vertical shift range.
        zoom_factor (float): Float representing zoom range.
        brightness_factor (float): Float representing brightness variation range.
        contrast_factor (float): Float representing contrast variation range.
        seed (int): Deterministic random seed.
        name (str): Model name.

    Returns:
        tf.keras.Sequential: Sequential augmentation model.
    """
    augmentation_layers = [layers.Input(shape=IMAGE_SHAPE, name="augmentation_input")]

    if horizontal_flip:
        augmentation_layers.append(
            layers.RandomFlip(mode="horizontal", seed=seed, name="random_horizontal_flip")
        )

    if rotation_factor > 0.0:
        augmentation_layers.append(
            layers.RandomRotation(factor=rotation_factor, seed=seed, name="random_rotation")
        )

    if width_shift_factor > 0.0 or height_shift_factor > 0.0:
        augmentation_layers.append(
            layers.RandomTranslation(
                height_factor=height_shift_factor,
                width_factor=width_shift_factor,
                seed=seed,
                name="random_translation",
            )
        )

    if zoom_factor > 0.0:
        augmentation_layers.append(
            layers.RandomZoom(height_factor=zoom_factor, seed=seed, name="random_zoom")
        )

    if brightness_factor > 0.0:
        augmentation_layers.append(
            layers.RandomBrightness(factor=brightness_factor, seed=seed, name="random_brightness")
        )

    if contrast_factor > 0.0:
        augmentation_layers.append(
            layers.RandomContrast(factor=contrast_factor, seed=seed, name="random_contrast")
        )

    pipeline = tf.keras.Sequential(augmentation_layers, name=name)
    return pipeline


# ------------------------------------------------------------------------------
# Augmented tf.data Dataset Pipeline Constructor
# ------------------------------------------------------------------------------
def create_light_augmented_train_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle_buffer: int = 10000,
    seed: int = RANDOM_SEED,
    augmentation_layer: Optional[tf.keras.layers.Layer] = None,
) -> tf.data.Dataset:
    """
    Construct high-performance tf.data.Dataset training pipeline with Light Data Augmentation.

    Pipeline Steps:
    1. from_tensor_slices((x_train, y_train))
    2. shuffle(buffer_size, seed=seed, reshuffle_each_iteration=True)
    3. batch(batch_size)
    4. map(augmentation_layer, num_parallel_calls=AUTOTUNE) -> applies horizontal flip per batch
    5. prefetch(AUTOTUNE)

    Args:
        x_train (np.ndarray): Training images of shape (N, 32, 32, 3) in [0.0, 1.0].
        y_train (np.ndarray): Training integer labels of shape (N,).
        batch_size (int): Mini-batch size. Defaults to DEFAULT_BATCH_SIZE (64).
        shuffle_buffer (int): Buffer size for training dataset shuffling. Defaults to 10,000.
        seed (int): Deterministic random seed. Defaults to RANDOM_SEED (42).
        augmentation_layer (Optional[tf.keras.layers.Layer]): Preprocessing layer/model.
            If None, instantiates get_light_augmentation_pipeline(seed=seed).

    Returns:
        tf.data.Dataset: Configured training tf.data.Dataset.
    """
    if augmentation_layer is None:
        augmentation_layer = get_light_augmentation_pipeline(seed=seed)

    logger.info(
        f"Creating light-augmented training tf.data pipeline (batch_size={batch_size}, shuffle_buffer={shuffle_buffer})..."
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_ds


def create_augmented_tf_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle_buffer: int = 10000,
    seed: int = RANDOM_SEED,
    augmentation_pipeline: Optional[tf.keras.layers.Layer] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Construct high-performance tf.data pipelines for Train (augmented), Val (unaugmented), and Test (unaugmented).

    Strict Experimental Controls:
    - Train: Shuffle + Batch + Light Augmentation (Horizontal Flip) + Prefetch.
    - Val: Batch + Prefetch (ZERO Augmentation).
    - Test: Batch + Prefetch (ZERO Augmentation, held strictly isolated).

    Args:
        x_train (np.ndarray): Training images.
        y_train (np.ndarray): Training labels.
        x_val (np.ndarray): Validation images.
        y_val (np.ndarray): Validation labels.
        x_test (np.ndarray): Test images.
        y_test (np.ndarray): Test labels.
        batch_size (int): Mini-batch size (default 64).
        shuffle_buffer (int): Shuffle buffer size for training (default 10,000).
        seed (int): Deterministic random seed.
        augmentation_pipeline (Optional[tf.keras.layers.Layer]): Augmentation layer for training set.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train_ds, val_ds, test_ds)
    """
    if augmentation_pipeline is None:
        augmentation_pipeline = get_light_augmentation_pipeline(seed=seed)

    logger.info(
        f"Building tf.data.Dataset pipelines: Train=Augmented(RandomFlip), Val=Unaugmented, Test=Unaugmented (batch_size={batch_size})..."
    )

    # Training pipeline: shuffle + batch + augment + prefetch
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .map(
            lambda x, y: (augmentation_pipeline(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Validation pipeline: batch + prefetch (UNMODIFIED / UNAUGMENTED)
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Test pipeline: batch + prefetch (UNMODIFIED / UNAUGMENTED / HELD ISOLATED)
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


# ------------------------------------------------------------------------------
# Augmentation Configuration Verification
# ------------------------------------------------------------------------------
def verify_light_augmentation_config(config: Dict[str, Any]) -> bool:
    """
    Strictly verify that an augmentation configuration dictionary complies with Experiment 2A controls.

    Guarantees:
    - horizontal_flip is True
    - rotation == 0
    - shift / translation == 0
    - zoom == 0
    - brightness == 0
    - contrast == 0
    - crop is False
    - validation and test are not augmented

    Args:
        config (Dict[str, Any]): Augmentation config section.

    Raises:
        ValueError: If any forbidden augmentation transform is enabled.

    Returns:
        bool: True if configuration complies with Experiment 2A controls.
    """
    if not config.get("horizontal_flip", False):
        raise ValueError("Experiment 2A violation: 'horizontal_flip' must be True.")

    disallowed_keys = {
        "rotation": 0.0,
        "rotation_factor": 0.0,
        "translation": 0.0,
        "shift": 0.0,
        "width_shift": 0.0,
        "height_shift": 0.0,
        "zoom": 0.0,
        "zoom_factor": 0.0,
        "brightness": 0.0,
        "brightness_factor": 0.0,
        "contrast": 0.0,
        "contrast_factor": 0.0,
    }

    for key, expected_val in disallowed_keys.items():
        if key in config and float(config[key]) != expected_val:
            raise ValueError(
                f"Experiment 2A violation: '{key}' must be {expected_val}, but found {config[key]}."
            )

    if config.get("crop", False):
        raise ValueError("Experiment 2A violation: 'crop' must be False.")

    if config.get("validation_augmented", False):
        raise ValueError("Experiment 2A violation: validation dataset must NOT be augmented.")

    if config.get("test_augmented", False):
        raise ValueError("Experiment 2A violation: test dataset must NOT be augmented.")

    return True


# ------------------------------------------------------------------------------
# Standalone Module Verification
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 75)
    print("  Transforms & Augmentation Module Verification (Experiment 2A)")
    print("=" * 75)

    pipeline = get_light_augmentation_pipeline(seed=RANDOM_SEED)
    print(f"  • Pipeline Name: {pipeline.name}")
    print(f"  • Layers: {[layer.name for layer in pipeline.layers]}")
    print(f"  • Total Layers: {len(pipeline.layers)}")

    # Test with dummy batch
    dummy_batch = tf.random.uniform(shape=(4, 32, 32, 3), minval=0.0, maxval=1.0)
    augmented_batch = pipeline(dummy_batch, training=True)
    unaugmented_batch = pipeline(dummy_batch, training=False)

    print(f"  • Input Shape: {dummy_batch.shape}")
    print(f"  • Augmented Output Shape (training=True): {augmented_batch.shape}")
    print(f"  • Inference Output Shape (training=False): {unaugmented_batch.shape}")
    assert augmented_batch.shape == dummy_batch.shape, "Shape mismatch after augmentation"
    assert unaugmented_batch.shape == dummy_batch.shape, "Shape mismatch during inference"

    print("  [SUCCESS] Transforms module verified successfully.")
    print("=" * 75)
