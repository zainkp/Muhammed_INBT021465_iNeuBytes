"""
CIFAR-10 Dataset Pipeline and Frozen Train/Validation/Test Split Manager.

This module provides deterministic data loading, strict split validation,
pixel normalization, and TensorFlow tf.data input pipeline construction for Task 1 (CIFAR-10).

Key Design & Safety Principles:
- Deterministic loading via tf.keras.datasets.cifar10.load_data().
- Stratified partitioning of the 50,000 training images into 40,000 train
  and 10,000 validation samples using the fixed project random seed.
- Complete isolation of the official 10,000 CIFAR-10 test samples for final evaluation.
- Persistent storage of frozen split indices in data/splits/ to ensure reproducibility.
- Automatic reuse of existing frozen split indices with strict validation
  (checks counts, seed, bounds, uniqueness, and non-overlap).
- Accidental overwrite protection: existing splits cannot be overwritten during normal execution.
- Clean normalization to [0.0, 1.0] (float32) without modifying raw source data.
- Baseline data pipeline without data augmentation.
"""

import json
import logging
import os
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
        CLASS_NAMES,
        IMAGE_SHAPE,
        NUM_CLASSES,
        RANDOM_SEED,
        SPLITS_DIR,
        TEST_SAMPLE_COUNT,
        TRAIN_SAMPLE_COUNT,
        VAL_SAMPLE_COUNT,
    )
except (ImportError, ModuleNotFoundError):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import (
        CLASS_NAMES,
        IMAGE_SHAPE,
        NUM_CLASSES,
        RANDOM_SEED,
        SPLITS_DIR,
        TEST_SAMPLE_COUNT,
        TRAIN_SAMPLE_COUNT,
        VAL_SAMPLE_COUNT,
    )

# Setup module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

# Default split manifest filename
DEFAULT_SPLIT_FILENAME = "cifar10_split_indices.json"
TOTAL_TRAINING_SOURCE_COUNT = 50_000


# ------------------------------------------------------------------------------
# Raw Data Loading
# ------------------------------------------------------------------------------
def load_cifar10_raw() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load raw, unmodified CIFAR-10 dataset using TensorFlow/Keras.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            ((x_train_full, y_train_full), (x_test, y_test)) where:
            - x_train_full: uint8 array of shape (50000, 32, 32, 3) in range [0, 255]
            - y_train_full: uint8 array of shape (50000, 1) in range [0, 9]
            - x_test: uint8 array of shape (10000, 32, 32, 3) in range [0, 255]
            - y_test: uint8 array of shape (10000, 1) in range [0, 9]
    """
    logger.info("Loading raw CIFAR-10 dataset via tf.keras.datasets.cifar10...")
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    logger.info(
        f"Raw data loaded: Train Pool={x_train_full.shape}, Test Set={x_test.shape}"
    )
    return (x_train_full, y_train_full), (x_test, y_test)


# ------------------------------------------------------------------------------
# Strict Split Validation
# ------------------------------------------------------------------------------
def validate_split_manifest(
    manifest: Dict[str, Any],
    expected_seed: int = RANDOM_SEED,
    expected_train_count: int = TRAIN_SAMPLE_COUNT,
    expected_val_count: int = VAL_SAMPLE_COUNT,
    total_source_count: int = TOTAL_TRAINING_SOURCE_COUNT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strictly validate the schema, metadata, and index integrity of a frozen split manifest.

    Validates:
    - Dataset is 'cifar10'
    - Split strategy is 'stratified'
    - Stored random seed matches the configured project seed
    - Exactly 40,000 train indices and 10,000 validation indices
    - Every index is within [0, 49,999]
    - No duplicate indices within train or validation sets
    - No overlap between train and validation indices (disjoint sets)

    Args:
        manifest (Dict[str, Any]): Loaded JSON manifest dictionary.
        expected_seed (int): Configured project random seed.
        expected_train_count (int): Expected training sample count (40,000).
        expected_val_count (int): Expected validation sample count (10,000).
        total_source_count (int): Expected total source sample count (50,000).

    Raises:
        ValueError: If any validation condition is violated.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Validated (train_indices, val_indices) as int64 arrays.
    """
    # 1. Validate dataset name
    dataset_name = manifest.get("dataset")
    if dataset_name != "cifar10":
        raise ValueError(
            f"Split validation error: dataset must be 'cifar10', found '{dataset_name}'."
        )

    # 2. Validate split strategy
    strategy = manifest.get("strategy")
    if strategy != "stratified":
        raise ValueError(
            f"Split validation error: strategy must be 'stratified', found '{strategy}'."
        )

    # 3. Validate random seed against project configuration
    stored_seed = manifest.get("random_seed")
    if stored_seed != expected_seed:
        raise ValueError(
            f"Split validation error: stored random seed ({stored_seed}) does not match "
            f"configured project seed ({expected_seed})."
        )

    # 4. Check presence of index keys
    if "train_indices" not in manifest or "val_indices" not in manifest:
        raise ValueError(
            "Split validation error: manifest missing 'train_indices' or 'val_indices' keys."
        )

    train_indices_raw = manifest["train_indices"]
    val_indices_raw = manifest["val_indices"]

    # 5. Validate exact partition counts
    if len(train_indices_raw) != expected_train_count:
        raise ValueError(
            f"Split validation error: expected exactly {expected_train_count} train indices, "
            f"found {len(train_indices_raw)}."
        )
    if len(val_indices_raw) != expected_val_count:
        raise ValueError(
            f"Split validation error: expected exactly {expected_val_count} validation indices, "
            f"found {len(val_indices_raw)}."
        )

    train_indices = np.array(train_indices_raw, dtype=np.int64)
    val_indices = np.array(val_indices_raw, dtype=np.int64)

    # 6. Validate index bounds [0, total_source_count - 1]
    if np.any(train_indices < 0) or np.any(train_indices >= total_source_count):
        raise ValueError(
            f"Split validation error: train indices contain values outside [0, {total_source_count - 1}]."
        )
    if np.any(val_indices < 0) or np.any(val_indices >= total_source_count):
        raise ValueError(
            f"Split validation error: validation indices contain values outside [0, {total_source_count - 1}]."
        )

    # 7. Validate uniqueness (no duplicates)
    train_set = set(train_indices.tolist())
    if len(train_set) != expected_train_count:
        raise ValueError(
            f"Split validation error: train indices contain duplicate entries. "
            f"Expected {expected_train_count} unique, found {len(train_set)}."
        )

    val_set = set(val_indices.tolist())
    if len(val_set) != expected_val_count:
        raise ValueError(
            f"Split validation error: validation indices contain duplicate entries. "
            f"Expected {expected_val_count} unique, found {len(val_set)}."
        )

    # 8. Validate non-overlapping partitions (zero data leakage)
    overlap = train_set.intersection(val_set)
    if len(overlap) > 0:
        raise ValueError(
            f"Split validation error: detected {len(overlap)} overlapping indices between "
            f"train and validation sets (data leakage violation)."
        )

    return train_indices, val_indices


# ------------------------------------------------------------------------------
# Split Generation & Frozen Split Management
# ------------------------------------------------------------------------------
def _generate_stratified_split_indices(
    labels: np.ndarray,
    train_count: int = TRAIN_SAMPLE_COUNT,
    val_count: int = VAL_SAMPLE_COUNT,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate deterministic stratified train and validation index partitions.

    Ensures perfectly balanced representation of each class in both sets.
    For CIFAR-10 (10 classes):
      - 4,000 samples per class for training (total 40,000)
      - 1,000 samples per class for validation (total 10,000)

    Args:
        labels (np.ndarray): Full training labels of shape (N, 1) or (N,).
        train_count (int): Desired number of train samples (default 40,000).
        val_count (int): Desired number of validation samples (default 10,000).
        seed (int): Fixed random seed for deterministic sampling.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (train_indices, val_indices) as int64 arrays.
    """
    labels_flat = labels.flatten()
    unique_classes, _ = np.unique(labels_flat, return_counts=True)
    num_classes = len(unique_classes)

    if (train_count + val_count) > len(labels_flat):
        raise ValueError(
            f"Requested train ({train_count}) + val ({val_count}) exceeds total samples ({len(labels_flat)})"
        )

    samples_per_class_train = train_count // num_classes
    samples_per_class_val = val_count // num_classes

    rng = np.random.default_rng(seed)

    train_indices_list: List[int] = []
    val_indices_list: List[int] = []

    for cls in unique_classes:
        cls_indices = np.where(labels_flat == cls)[0]
        # Deterministically permute the indices belonging to this class
        permuted_indices = rng.permutation(cls_indices)

        cls_train = permuted_indices[:samples_per_class_train]
        cls_val = permuted_indices[
            samples_per_class_train : samples_per_class_train + samples_per_class_val
        ]

        train_indices_list.extend(cls_train.tolist())
        val_indices_list.extend(cls_val.tolist())

    train_indices = np.array(sorted(train_indices_list), dtype=np.int64)
    val_indices = np.array(sorted(val_indices_list), dtype=np.int64)

    logger.info(
        f"Generated stratified split: Train={len(train_indices)} samples, Val={len(val_indices)} samples "
        f"({samples_per_class_train} train / {samples_per_class_val} val per class across {num_classes} classes)."
    )
    return train_indices, val_indices


def get_or_create_frozen_splits(
    y_train_full: np.ndarray,
    splits_dir: Union[str, Path] = SPLITS_DIR,
    split_filename: str = DEFAULT_SPLIT_FILENAME,
    train_count: int = TRAIN_SAMPLE_COUNT,
    val_count: int = VAL_SAMPLE_COUNT,
    seed: int = RANDOM_SEED,
    allow_overwrite: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Retrieve or create persistent frozen train/validation split indices.

    Safety & Reproducibility Guarantees:
    - If the split file exists, it is loaded, strictly validated, and reused.
    - Overwriting an existing split file is strictly disallowed during normal execution.
      Regeneration requires passing allow_overwrite=True explicitly.
    - If validation fails on an existing file, a ValueError is raised.

    Args:
        y_train_full (np.ndarray): Array of labels for the full training pool (50,000).
        splits_dir (Union[str, Path]): Directory where split indices are stored.
        split_filename (str): JSON filename for storing split indices.
        train_count (int): Total training samples (default 40,000).
        val_count (int): Total validation samples (default 10,000).
        seed (int): Fixed random seed matching project configuration.
        allow_overwrite (bool): Explicit safety flag to allow overwriting an existing split.
                                Defaults to False.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing validated 'train_indices' and 'val_indices'.
    """
    splits_path = Path(splits_dir)
    split_file = splits_path / split_filename

    # If file exists and overwrite is not explicitly authorized, reuse and validate
    if split_file.exists() and not allow_overwrite:
        logger.info(f"Loading and validating existing frozen split manifest from: {split_file}")
        try:
            with open(split_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read split manifest JSON at {split_file}: {e}") from e

        train_indices, val_indices = validate_split_manifest(
            manifest=manifest,
            expected_seed=seed,
            expected_train_count=train_count,
            expected_val_count=val_count,
            total_source_count=len(y_train_full),
        )

        logger.info(
            f"Frozen split verified & loaded successfully: {len(train_indices)} Train, {len(val_indices)} Val."
        )
        return {"train_indices": train_indices, "val_indices": val_indices}

    # If file exists and allow_overwrite is False, this path cannot be reached.
    # If file does not exist, or allow_overwrite is explicitly True, generate a new split:
    logger.info(
        f"Generating deterministic stratified split: Train={train_count}, Val={val_count}, Seed={seed}..."
    )
    train_indices, val_indices = _generate_stratified_split_indices(
        labels=y_train_full,
        train_count=train_count,
        val_count=val_count,
        seed=seed,
    )

    # Save to disk
    splits_path.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "dataset": "cifar10",
        "random_seed": seed,
        "strategy": "stratified",
        "num_total_training_source": int(len(y_train_full)),
        "num_train_samples": int(len(train_indices)),
        "num_val_samples": int(len(val_indices)),
        "num_test_samples": int(TEST_SAMPLE_COUNT),
        "class_names": CLASS_NAMES,
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
    }

    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    logger.info(f"Saved frozen split manifest to: {split_file}")
    return {"train_indices": train_indices, "val_indices": val_indices}


# ------------------------------------------------------------------------------
# Normalization & Dataset Splitting
# ------------------------------------------------------------------------------
def normalize_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values from [0, 255] uint8 to [0.0, 1.0] float32.

    The original input array is left unmodified.

    Args:
        images (np.ndarray): Image array with values in [0, 255].

    Returns:
        np.ndarray: Normalized float32 image array with values in [0.0, 1.0].
    """
    return images.astype(np.float32) / 255.0


def load_cifar10_data(
    normalize: bool = True,
    flatten_labels: bool = True,
    splits_dir: Union[str, Path] = SPLITS_DIR,
    split_filename: str = DEFAULT_SPLIT_FILENAME,
    seed: int = RANDOM_SEED,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Complete data pipeline for CIFAR-10: loads data, applies validated frozen split, and normalizes.

    Guarantees:
    - 40,000 Train samples (stratified, balanced across 10 classes)
    - 10,000 Validation samples (stratified, balanced across 10 classes)
    - 10,000 Test samples (original CIFAR-10 test set, reserved exclusively for test evaluation)
    - No data augmentation applied.
    - Zero data leakage between partitions.

    Args:
        normalize (bool): If True, normalizes pixel values to [0.0, 1.0] (float32).
        flatten_labels (bool): If True, flattens label vectors from (N, 1) to (N,) (int64).
        splits_dir (Union[str, Path]): Directory where split indices are stored.
        split_filename (str): Name of split JSON file.
        seed (int): Fixed random seed for split validation / generation.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    """
    # 1. Load raw data
    (x_train_full, y_train_full), (x_test_raw, y_test_raw) = load_cifar10_raw()

    # 2. Retrieve and validate deterministic frozen split
    split_dict = get_or_create_frozen_splits(
        y_train_full=y_train_full,
        splits_dir=splits_dir,
        split_filename=split_filename,
        train_count=TRAIN_SAMPLE_COUNT,
        val_count=VAL_SAMPLE_COUNT,
        seed=seed,
        allow_overwrite=False,
    )
    train_idx = split_dict["train_indices"]
    val_idx = split_dict["val_indices"]

    # 3. Partition datasets
    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]

    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    x_test = x_test_raw
    y_test = y_test_raw

    # 4. Normalize pixel values if requested (leaves raw data unmodified)
    if normalize:
        logger.info("Normalizing image pixel values to [0.0, 1.0] (float32)...")
        x_train = normalize_images(x_train)
        x_val = normalize_images(x_val)
        x_test = normalize_images(x_test)

    # 5. Format labels
    if flatten_labels:
        y_train = y_train.flatten().astype(np.int64)
        y_val = y_val.flatten().astype(np.int64)
        y_test = y_test.flatten().astype(np.int64)
    else:
        y_train = y_train.astype(np.int64)
        y_val = y_val.astype(np.int64)
        y_test = y_test.astype(np.int64)

    logger.info(
        f"CIFAR-10 Partitioning Complete:\n"
        f"  - Train Set     : Images={x_train.shape} (dtype={x_train.dtype}), Labels={y_train.shape}\n"
        f"  - Validation Set: Images={x_val.shape} (dtype={x_val.dtype}), Labels={y_val.shape}\n"
        f"  - Test Set      : Images={x_test.shape} (dtype={x_test.dtype}), Labels={y_test.shape}"
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ------------------------------------------------------------------------------
# tf.data Pipeline Builder (Baseline - No Augmentation)
# ------------------------------------------------------------------------------
def create_tf_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    shuffle_buffer: int = 10000,
    seed: int = RANDOM_SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Construct high-performance tf.data.Dataset pipelines for Train, Val, and Test splits.

    Baseline pipeline specifications:
    - Train: Deterministic shuffling with fixed seed + Batching + Prefetching (AUTOTUNE).
      No data augmentation.
    - Val: Batching + Prefetching (AUTOTUNE).
    - Test: Batching + Prefetching (AUTOTUNE).

    Args:
        x_train (np.ndarray): Training images.
        y_train (np.ndarray): Training labels.
        x_val (np.ndarray): Validation images.
        y_val (np.ndarray): Validation labels.
        x_test (np.ndarray): Test images.
        y_test (np.ndarray): Test labels.
        batch_size (int): Batch size for batching (default 64).
        shuffle_buffer (int): Buffer size for training dataset shuffling (default 10,000).
        seed (int): Random seed for reproducible dataset shuffling.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train_ds, val_ds, test_ds)
    """
    logger.info(f"Building tf.data.Dataset pipelines with batch_size={batch_size}...")

    # Training pipeline: shuffle + batch + prefetch
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Validation pipeline: batch + prefetch
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Test pipeline: batch + prefetch
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


# ------------------------------------------------------------------------------
# Dataset Summary & Shape Inspection
# ------------------------------------------------------------------------------
def get_dataset_summary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """
    Generate a comprehensive structural and statistical summary of the dataset splits.

    Args:
        x_train (np.ndarray): Training images.
        y_train (np.ndarray): Training labels.
        x_val (np.ndarray): Validation images.
        y_val (np.ndarray): Validation labels.
        x_test (np.ndarray): Test images.
        y_test (np.ndarray): Test labels.
        class_names (List[str]): List of categorical class name strings.

    Returns:
        Dict[str, Any]: Detailed dictionary of dataset metrics, shapes, and class counts.
    """
    y_train_flat = y_train.flatten()
    y_val_flat = y_val.flatten()
    y_test_flat = y_test.flatten()

    def _class_distribution(labels: np.ndarray) -> Dict[str, int]:
        unique, counts = np.unique(labels, return_counts=True)
        dist = {}
        for u, c in zip(unique, counts):
            name = class_names[int(u)] if int(u) < len(class_names) else f"class_{u}"
            dist[name] = int(c)
        return dist

    summary = {
        "dataset_name": "CIFAR-10",
        "num_classes": len(class_names),
        "class_names": class_names,
        "splits": {
            "train": {
                "num_samples": int(x_train.shape[0]),
                "image_shape": list(x_train.shape[1:]),
                "image_dtype": str(x_train.dtype),
                "pixel_min": float(np.min(x_train)),
                "pixel_max": float(np.max(x_train)),
                "class_distribution": _class_distribution(y_train_flat),
            },
            "validation": {
                "num_samples": int(x_val.shape[0]),
                "image_shape": list(x_val.shape[1:]),
                "image_dtype": str(x_val.dtype),
                "pixel_min": float(np.min(x_val)),
                "pixel_max": float(np.max(x_val)),
                "class_distribution": _class_distribution(y_val_flat),
            },
            "test": {
                "num_samples": int(x_test.shape[0]),
                "image_shape": list(x_test.shape[1:]),
                "image_dtype": str(x_test.dtype),
                "pixel_min": float(np.min(x_test)),
                "pixel_max": float(np.max(x_test)),
                "class_distribution": _class_distribution(y_test_flat),
            },
        },
    }
    return summary


def print_dataset_summary(summary: Dict[str, Any]) -> None:
    """
    Print a formatted, human-readable summary of the dataset splits and class distribution.

    Args:
        summary (Dict[str, Any]): Summary dictionary from get_dataset_summary().
    """
    print("=" * 70)
    print(f"  Dataset: {summary['dataset_name']} ({summary['num_classes']} Classes)")
    print("=" * 70)
    for split_name, details in summary["splits"].items():
        print(f"\n📁 [{split_name.upper()} SET]")
        print(f"   • Samples     : {details['num_samples']:,}")
        print(f"   • Image Shape : {details['image_shape']} ({details['image_dtype']})")
        print(f"   • Value Range : [{details['pixel_min']:.2f}, {details['pixel_max']:.2f}]")
        print("   • Class Balance:")
        for cls_name, count in details["class_distribution"].items():
            print(f"       - {cls_name:<12}: {count:,} samples")
    print("=" * 70)


# ------------------------------------------------------------------------------
# CIFAR-10 Data Manager Class (High-Level Interface)
# ------------------------------------------------------------------------------
class CIFAR10Pipeline:
    """
    High-level data pipeline manager for CIFAR-10.

    Provides a clean object-oriented interface for loading data, retrieving
    strictly validated frozen splits, building tf.data pipelines, and inspecting shapes.
    """

    def __init__(
        self,
        splits_dir: Union[str, Path] = SPLITS_DIR,
        split_filename: str = DEFAULT_SPLIT_FILENAME,
        seed: int = RANDOM_SEED,
        normalize: bool = True,
        flatten_labels: bool = True,
    ) -> None:
        """
        Initialize CIFAR10Pipeline with experiment settings.

        Args:
            splits_dir (Union[str, Path]): Directory for frozen split files.
            split_filename (str): Filename for split manifest.
            seed (int): Fixed random seed.
            normalize (bool): Whether to normalize images to [0.0, 1.0].
            flatten_labels (bool): Whether to flatten label vectors to 1D.
        """
        self.splits_dir = Path(splits_dir)
        self.split_filename = split_filename
        self.seed = seed
        self.normalize = normalize
        self.flatten_labels = flatten_labels

        self.class_names: List[str] = CLASS_NAMES
        self.num_classes: int = NUM_CLASSES
        self.image_shape: Tuple[int, int, int] = IMAGE_SHAPE

        self._data: Optional[
            Tuple[
                Tuple[np.ndarray, np.ndarray],
                Tuple[np.ndarray, np.ndarray],
                Tuple[np.ndarray, np.ndarray],
            ]
        ] = None

    def load(
        self,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        Load and partition the CIFAR-10 dataset using the validated frozen split.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
                ((x_train, y_train), (x_val, y_val), (x_test, y_test))
        """
        self._data = load_cifar10_data(
            normalize=self.normalize,
            flatten_labels=self.flatten_labels,
            splits_dir=self.splits_dir,
            split_filename=self.split_filename,
            seed=self.seed,
        )
        return self._data

    def get_numpy_data(
        self,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        Retrieve numpy arrays for (train, val, test) splits. Loads data if not already loaded.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
                ((x_train, y_train), (x_val, y_val), (x_test, y_test))
        """
        if self._data is None:
            return self.load()
        return self._data

    def get_tf_datasets(
        self,
        batch_size: int = 64,
        shuffle_buffer: int = 10000,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Build tf.data.Dataset pipelines for train, val, and test splits.

        Args:
            batch_size (int): Mini-batch size.
            shuffle_buffer (int): Shuffle buffer size for training set.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train_ds, val_ds, test_ds)
        """
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.get_numpy_data()
        return create_tf_datasets(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            seed=self.seed,
        )

    def summary(self) -> Dict[str, Any]:
        """
        Get dataset summary dictionary.

        Returns:
            Dict[str, Any]: Summary dictionary.
        """
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.get_numpy_data()
        return get_dataset_summary(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            class_names=self.class_names,
        )

    def print_summary(self) -> None:
        """Print formatted dataset summary."""
        summary_dict = self.summary()
        print_dataset_summary(summary_dict)
