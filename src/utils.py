import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    """Sets random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[INFO] Random seed set to: {seed}")
