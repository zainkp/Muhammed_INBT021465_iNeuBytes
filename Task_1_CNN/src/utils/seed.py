"""
Reproducibility utilities for setting deterministic random seeds across Python, NumPy, and TensorFlow.
"""

import os
import random
import numpy as np

def set_seed(seed: int = 42) -> None:
    """
    Set seeds across standard library, NumPy, and TensorFlow to ensure reproducible runs.
    
    Args:
        seed (int): The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
