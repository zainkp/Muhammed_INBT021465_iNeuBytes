"""
Baseline Convolutional Neural Network (CNN) Architecture for CIFAR-10.

This module implements an adapted AlexNet-style Convolutional Neural Network
specifically configured for 32x32 RGB images from the CIFAR-10 dataset.

Architecture Overview:
- Stage 1 (Block 1): 2x Conv2D (64 filters, 3x3 kernel, ReLU, 'same' padding) + MaxPooling2D (2x2, stride 2)
- Stage 2 (Block 2): 2x Conv2D (128 filters, 3x3 kernel, ReLU, 'same' padding) + MaxPooling2D (2x2, stride 2)
- Stage 3 (Block 3): 1x Conv2D (256 filters, 3x3 kernel, ReLU, 'same' padding) + MaxPooling2D (2x2, stride 2)
- Stage 4 (Classifier Head): Flatten + Dense (512 units, ReLU) + Output Dense (10 units, Softmax)

Baseline Design Guarantees & Constraints:
- Pure baseline architecture without regularization (No Dropout, No L1/L2 weight decay).
- No Batch Normalization layers.
- No Data Augmentation.
- No Pretrained models or Transfer Learning backbones.
- Standalone uncompiled model returned (compilation reserved for training phase).
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------------------------------------------------------------------
# Config & Path Handling (Support both package and direct execution)
# ------------------------------------------------------------------------------
try:
    from src.config import IMAGE_SHAPE, NUM_CLASSES
except (ImportError, ModuleNotFoundError):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import IMAGE_SHAPE, NUM_CLASSES


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = IMAGE_SHAPE,
    num_classes: int = NUM_CLASSES,
    name: str = "baseline_cifar10_cnn",
) -> tf.keras.Model:
    """
    Build an uncompiled adapted AlexNet-style Baseline CNN for CIFAR-10 classification.

    The architecture adapts classic AlexNet hierarchical convolutional feature extraction
    for 32x32 image inputs using compact 3x3 receptive fields with 'same' spatial padding,
    followed by progressive 2x2 max-pooling and a fully connected classification head.

    Baseline Restrictions Enforced:
    - No Dropout layers.
    - No Batch Normalization layers.
    - No L1 / L2 weight regularization.
    - Uncompiled model returned (loss function and optimizer configured in training module).

    Args:
        input_shape (Tuple[int, int, int]): Dimensions of input images (H, W, C). Defaults to (32, 32, 3).
        num_classes (int): Number of target classification classes. Defaults to 10.
        name (str): Name identifier for the Keras model. Defaults to "baseline_cifar10_cnn".

    Returns:
        tf.keras.Model: Uncompiled Keras Sequential model instance.
    """
    model = models.Sequential(
        [
            # Input Specification
            layers.Input(shape=input_shape, name="input_layer"),
            # ------------------------------------------------------------------
            # Stage 1: Feature Extraction Block 1 (Output: 16x16x64)
            # ------------------------------------------------------------------
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv1_1",
            ),
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv1_2",
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="pool1",
            ),
            # ------------------------------------------------------------------
            # Stage 2: Feature Extraction Block 2 (Output: 8x8x128)
            # ------------------------------------------------------------------
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv2_1",
            ),
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv2_2",
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="pool2",
            ),
            # ------------------------------------------------------------------
            # Stage 3: Feature Extraction Block 3 (Output: 4x4x256)
            # ------------------------------------------------------------------
            layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv3_1",
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="pool3",
            ),
            # ------------------------------------------------------------------
            # Stage 4: Dense Classifier Head (Output: 10)
            # ------------------------------------------------------------------
            layers.Flatten(name="flatten"),
            layers.Dense(
                units=512,
                activation="relu",
                name="fc1",
            ),
            layers.Dense(
                units=num_classes,
                activation="softmax",
                name="output_dense",
            ),
        ],
        name=name,
    )

    return model


def get_model_summary(model: Optional[tf.keras.Model] = None) -> None:
    """
    Print the detailed summary and parameter count of the baseline CNN architecture.

    Args:
        model (Optional[tf.keras.Model]): Model to inspect. If None, builds default baseline CNN.
    """
    if model is None:
        model = build_baseline_cnn()
    model.summary()


if __name__ == "__main__":
    print("=" * 75)
    print("  CIFAR-10 Adapted AlexNet-Style Baseline CNN Architecture")
    print("=" * 75)
    baseline_model = build_baseline_cnn()
    get_model_summary(baseline_model)

    print("\n" + "=" * 75)
    print("  Baseline Constraints Verification")
    print("=" * 75)
    layer_types = [type(layer).__name__ for layer in baseline_model.layers]
    has_dropout = any("dropout" in t.lower() for t in layer_types)
    has_batchnorm = any("batchnorm" in t.lower() or "batchnormalization" in t.lower() for t in layer_types)
    has_regularizer = any(
        getattr(layer, "kernel_regularizer", None) is not None
        or getattr(layer, "bias_regularizer", None) is not None
        or getattr(layer, "activity_regularizer", None) is not None
        for layer in baseline_model.layers
    )

    print(f"  - Compiled Status         : {'Compiled' if baseline_model.compiled else 'Uncompiled (Correct)'}")
    print(f"  - Total Layers            : {len(baseline_model.layers)}")
    print(f"  - Total Parameters        : {baseline_model.count_params():,}")
    print(f"  - Trainable Parameters    : {sum(tf.size(w).numpy() for w in baseline_model.trainable_weights):,}")
    print(f"  - Non-Trainable Parameters: {sum(tf.size(w).numpy() for w in baseline_model.non_trainable_weights):,}")
    print(f"  - Dropout Present         : {'YES (Violates Baseline)' if has_dropout else 'NO (Baseline Compliant)'}")
    print(f"  - BatchNorm Present       : {'YES (Violates Baseline)' if has_batchnorm else 'NO (Baseline Compliant)'}")
    print(f"  - Regularizers Present    : {'YES (Violates Baseline)' if has_regularizer else 'NO (Baseline Compliant)'}")
    print("=" * 75)
