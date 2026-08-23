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