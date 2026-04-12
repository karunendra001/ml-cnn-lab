import tensorflow as tf
import numpy as np
import random
import pandas as pd
import matplotlib

# -----------------------------
# Set seeds for reproducibility
# -----------------------------
SEED = 42

# Controls Python random operations
random.seed(SEED)

# Controls NumPy random operations
np.random.seed(SEED)

# Controls TensorFlow randomness
tf.random.set_seed(SEED)

# -----------------------------
# Print library versions
# -----------------------------
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", matplotlib.__version__)

# -----------------------------
# Check GPU availability
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("\nGPU is available!")
    print("GPU Devices:", gpus)
else:
    print("\nNo GPU found.")

    # Explanation required by assignment
    print("""
    NOTE:
    CPU training is slower because it processes operations sequentially,
    while GPUs perform parallel computations using thousands of cores.

    On a GPU machine:
    - Training would be much faster
    - Larger models can be trained efficiently
    - Batch processing is optimized
    """)