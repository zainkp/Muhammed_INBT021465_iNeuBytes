"""
Execution Entry Point for Task 1 Part B — Experiment Group 3: SGD + Momentum (LR=0.01).

Configuration:
- Model: Pure Baseline CNN via build_baseline_cnn() (2,658,122 parameters)
- Optimizer: SGD + Momentum (0.9), learning_rate=0.01, nesterov=False
- Epochs: 30 (Fixed)
- Batch Size: 64
- Split: Frozen 40k train / 10k val / 10k isolated test
- Seed: 42

Usage:
    python Task_1_CNN/src/training/run_opt_sgd_lr_01.py
    python -m src.training.run_opt_sgd_lr_01
"""

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
TASK_ROOT = CURRENT_FILE.parent.parent.parent
REPO_ROOT = TASK_ROOT.parent

for path in [str(TASK_ROOT), str(REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.training.run_optimization_study import run_single_optimization_experiment

if __name__ == "__main__":
    run_single_optimization_experiment("opt_sgd_lr_01")
