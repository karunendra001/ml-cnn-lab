import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = tf.keras.models.load_model("model.h5")

# Find first Conv layer
conv_layer = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        conv_layer = layer
        break

# Get filters
filters, _ = conv_layer.get_weights()

n_filters = filters.shape[-1]

plt.figure(figsize=(8,8))

for i in range(n_filters):
    f = filters[:, :, :, i]

    # Normalize each filter
    f_min, f_max = f.min(), f.max()
    if f_max != f_min:
        f = (f - f_min) / (f_max - f_min)

    plt.subplot(6,6,i+1)
    plt.imshow(f)
    plt.title(str(i))
    plt.axis('off')

plt.tight_layout()
plt.savefig("conv1_filters.png")
plt.show()