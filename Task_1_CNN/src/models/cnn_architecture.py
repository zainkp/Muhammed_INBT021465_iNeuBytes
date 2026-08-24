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
    Build and return the uncompiled baseline CNN architecture for CIFAR-10 classification.

    Architecture:
        - Input: input_shape (default: 32x32x3)
        - Block 1:
            - Conv2D(64, (3, 3), padding='same', activation='relu')
            - Conv2D(64, (3, 3), padding='same', activation='relu')
            - MaxPooling2D(pool_size=(2, 2), strides=2)
        - Block 2:
            - Conv2D(128, (3, 3), padding='same', activation='relu')
            - Conv2D(128, (3, 3), padding='same', activation='relu')
            - MaxPooling2D(pool_size=(2, 2), strides=2)
        - Block 3:
            - Conv2D(256, (3, 3), padding='same', activation='relu')
            - MaxPooling2D(pool_size=(2, 2), strides=2)
        - Classification Head:
            - Flatten()
            - Dense(512, activation='relu')
            - Dense(num_classes, activation='softmax')

    Args:
        input_shape (Tuple[int, int, int]): Dimensions of input images (H, W, C). Defaults to (32, 32, 3).
        num_classes (int): Number of target classification classes. Defaults to 10.
        name (str): Name of the Keras model. Defaults to "baseline_cifar10_cnn".

    Returns:
        tf.keras.Model: An uncompiled Keras Sequential model instance.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_layer"),
            # Stage 1: Block 1
            layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv1_1"),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv1_2"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool1"),
            # Stage 2: Block 2
            layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2_1"),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2_2"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool2"),
            # Stage 3: Block 3
            layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3_1"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool3"),
            # Stage 4: Classification Head
            layers.Flatten(name="flatten"),
            layers.Dense(512, activation="relu", name="dense1"),
            layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name=name,
    )
    return model


def get_model_summary(model: Optional[tf.keras.Model] = None) -> str:
    """
    Generate and return a formatted string summary of the CNN architecture.

    Args:
        model (Optional[tf.keras.Model]): Keras model instance to summarize.
            If None, a new baseline CNN model will be constructed.

    Returns:
        str: Detailed string summary of the model architecture.
    """
    if model is None:
        model = build_baseline_cnn()
    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    return "\n".join(summary_lines)


if __name__ == "__main__":
    print("=" * 70)
    print("  Baseline CIFAR-10 CNN Architecture Verification")
    print("=" * 70)

    # 1. Build Model
    model = build_baseline_cnn()

    # 2. Print Summary and Shapes
    print(f"\nModel Name: {model.name}")
    print(f"Input Shape: {model.input_shape}")
    print(f"Output Shape: {model.output_shape}")
    print(f"Total Parameters: {model.count_params():,}")
    print(f"Trainable Parameters: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    print(f"Non-trainable Parameters: {sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights):,}")
    print(f"Is Compiled: {model.compiled}")

    # 3. Model Architecture Summary
    print("\n--- Model Summary ---")
    print(get_model_summary(model))

    # 4. Confirm uncompiled state
    assert not model.compiled, "Baseline model should not be compiled upon construction."
    assert getattr(model, "optimizer", None) is None, "Baseline model optimizer should be None before trainer compilation."
    print("\n[SUCCESS] Baseline CNN architecture constructed and verified successfully.")