# Task 1: Computer Vision using CNN Models (CIFAR-10)
**iNeuBytes Artificial Intelligence Internship**

---

## 📌 Project Overview
This repository contains the end-to-end implementation for **Task 1: Computer Vision using Convolutional Neural Networks (CNNs)** for the iNeuBytes AI Internship program. 

The focus of Task 1 is training and evaluating an **AlexNet-style CNN architecture adapted for 32×32 RGB images** on the **CIFAR-10** dataset using **TensorFlow/Keras**. All experiments follow rigorous reproducibility standards with a **frozen train/validation/test split** and a **fixed epoch budget** per experiment run.

---

## 📁 Project Structure

```text
Task_1_CNN/
│
├── data/                                # Dataset storage and split definitions
│   ├── raw/                             # Original, immutable CIFAR-10 data batches
│   ├── processed/                       # Preprocessed / normalized datasets
│   └── splits/                          # Frozen train/val/test split manifests & indices
│
├── experiments/                         # Prototyping & experiment management
│   ├── configs/                         # YAML experiment configurations (hyperparameters)
│   │   └── baseline_config.yaml         # Baseline AlexNet-style CNN configuration
│   └── notebooks/                       # Jupyter notebooks for EDA and exploratory runs
│
├── figures/                             # Generated figures & visual deliverables
│   └── .gitkeep                         # Loss curves, accuracy curves, confusion matrix heatmaps
│
├── models/                              # Serialized model artifacts (TensorFlow/Keras)
│   ├── checkpoints/                     # Epoch checkpoint models (.keras / .h5)
│   └── exported/                        # Final exported models for inference
│
├── results/                             # Quantitative evaluation artifacts
│   ├── logs/                            # CSV logs and TensorBoard training histories
│   └── metrics/                         # Classification reports, per-class metrics & summary tables
│
├── src/                                 # Modular source code
│   ├── data/                            # CIFAR-10 loading, split partitioning, tf.data pipelines
│   │   ├── __init__.py
│   │   ├── dataset.py                   # Data loader with frozen split manager
│   │   └── transforms.py                # Preprocessing & augmentation pipelines
│   ├── models/                          # AlexNet-style CNN architecture (adapted for 32x32)
│   │   ├── __init__.py
│   │   └── cnn_architecture.py         # Keras Model subclass / Functional model definition
│   ├── training/                        # Training loop and evaluation routines
│   │   ├── __init__.py
│   │   ├── trainer.py                   # Trainer engine with fixed-epoch schedule & checkpointing
│   │   └── evaluator.py                 # Evaluation engine for validation and test sets
│   ├── utils/                           # Helper utilities
│   │   ├── __init__.py
│   │   ├── metrics.py                   # Accuracy, Precision, Recall, F1 score computation
│   │   ├── plotting.py                  # Loss curves, accuracy curves, confusion matrix plots
│   │   └── seed.py                      # Global deterministic seeding for TensorFlow/NumPy
│   ├── __init__.py
│   └── config.py                        # Path constants, CIFAR-10 specs, default hyperparameters
│
├── .gitignore                           # Git ignore rules for data, checkpoints, and cache
├── requirements.txt                     # TensorFlow/Keras dependencies
└── README.md                            # Project documentation (this file)
```

---

## ⚙️ Environment Setup & Installation

### 1. Create a Virtual Environment
```bash
# Using Python venv
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔬 Experimentation & Protocol Guidelines

1. **Dataset & Frozen Splits:**
   - Dataset: **CIFAR-10** (10 classes: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*).
   - Image Dimensions: **32×32×3** (RGB).
   - Split Strategy: Partitioned into deterministic **Train (40,000)**, **Validation (10,000)**, and **Test (10,000)** sets. The exact sample indices are frozen and stored in `data/splits/split_indices.json` to guarantee identical evaluation subsets across all experiments.

2. **Model Architecture:**
   - **Adapted AlexNet-style CNN**: Scaled convolutional filter sizes, strides, and pooling windows tailored for 32×32 input resolution, preventing excessive spatial downsampling while preserving multi-stage feature extraction, batch normalization / local response features, dropout regularization, and fully connected classification layers.

3. **Training Protocol:**
   - **Fixed Epoch Budget**: Experiments are trained across a predetermined fixed number of epochs (no default early stopping) to ensure fair and consistent cross-experiment comparisons unless an experiment specifically studies epoch dynamics.
   - **Checkpointing**: Best model weights per run are saved in `models/checkpoints/` using the `.keras` format based on validation performance.

4. **Deliverables & Evaluation:**
   - **Curves:** Training vs. Validation Loss curves and Accuracy curves saved in `figures/`.
   - **Classification Metrics:** Overall accuracy, macro-averaged and weighted precision, recall, and F1-score saved in `results/metrics/`.
   - **Confusion Matrix:** Categorical confusion matrix heatmap saved in `figures/`.
