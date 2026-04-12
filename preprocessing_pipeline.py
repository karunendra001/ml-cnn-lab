import tensorflow as tf
import numpy as np

# -----------------------------
# Load MNIST
# -----------------------------
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# Take a small sample to show before/after
x_sample = x_train[:5]
y_sample = y_train[:5]

print("===== BEFORE PROCESSING =====")
print("Image shape:", x_sample.shape)
print("Image dtype:", x_sample.dtype)
print("Pixel range:", x_sample.min(), "to", x_sample.max())
print("Labels:", y_sample)

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess(x, y):
    # (a) Normalize (IMPORTANT: use 255.0)
    x = x.astype('float32') / 255.0

    # (b) Reshape (add channel dimension)
    x = x.reshape(-1, 28, 28, 1)

    # (c) One-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=10)

    return x, y

# Apply preprocessing
x_processed, y_processed = preprocess(x_sample, y_sample)

# -----------------------------
# AFTER PROCESSING
# -----------------------------
print("\n===== AFTER PROCESSING =====")
print("Image shape:", x_processed.shape)
print("Image dtype:", x_processed.dtype)
print("Pixel range:", x_processed.min(), "to", x_processed.max())

print("Labels shape:", y_processed.shape)
print("Sample one-hot label:", y_processed[0])