import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load CIFAR-10
# -----------------------------
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

# Normalize (good practice)
x_train = x_train.astype('float32') / 255.0

# -----------------------------
# Augmentation Pipeline
# -----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),     # (a)
    tf.keras.layers.RandomRotation(0.1),          # (b) ~ ±10 degrees
    tf.keras.layers.RandomZoom(0.1)               # (c) up to 10%
])

# -----------------------------
# Select 5 images
# -----------------------------
sample_images = x_train[:5]

# -----------------------------
# Plot (5x4 grid)
# -----------------------------
fig, axes = plt.subplots(5, 4, figsize=(10, 10))

for i in range(5):
    # Original
    axes[i, 0].imshow(sample_images[i])
    axes[i, 0].set_title("Original")
    axes[i, 0].axis('off')

    # 3 Augmented versions
    for j in range(1, 4):
        augmented = data_augmentation(tf.expand_dims(sample_images[i], 0))
        axes[i, j].imshow(augmented[0])
        axes[i, j].set_title("Augmented")
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig("augmentation_demo.png")
plt.show()