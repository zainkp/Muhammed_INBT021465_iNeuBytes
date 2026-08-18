"""
Configuration module for Task 1: Computer Vision using CNN (CIFAR-10).
Contains path definitions, dataset specifications, and default hyperparameters.
"""

from pathlib import Path

# ==============================================================================
# Directory Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Experiment Paths
EXPERIMENTS_DIR = BASE_DIR / "experiments"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
NOTEBOOKS_DIR = EXPERIMENTS_DIR / "notebooks"

# Model Paths
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTED_MODELS_DIR = MODELS_DIR / "exported"

# Results & Output Paths
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = BASE_DIR / "figures"

# ==============================================================================
# CIFAR-10 Dataset Specifications
# ==============================================================================
DATASET_NAME = "cifar10"
IMAGE_SHAPE = (32, 32, 3)
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Standard Frozen Split Sizes (50,000 total training -> 40,000 train / 10,000 val)
TRAIN_SAMPLE_COUNT = 40_000
VAL_SAMPLE_COUNT = 10_000
TEST_SAMPLE_COUNT = 10_000

# ==============================================================================
# Default Training Protocol (Fixed Epoch Budget)
# ==============================================================================
RANDOM_SEED = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 30  # Fixed epoch budget across experiments
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.4
DEFAULT_WEIGHT_DECAY = 1e-4
