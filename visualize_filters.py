import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load your trained model
# -----------------------------
model = tf.keras.models.load_model("custom_cnn_final.h5")  # change if needed

# -----------------------------
# Get first Conv layer weights
# -----------------------------
filters, biases = model.layers[1].get_weights()  
# (skip Input layer → Conv2D is usually index 1)

print("Filter shape:", filters.shape)

# -----------------------------
# Normalize filters to [0,1]
# -----------------------------
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# -----------------------------
# Plot filters
# -----------------------------
n_filters = filters.shape[-1]

plt.figure(figsize=(10,10))

for i in range(n_filters):
    f = filters[:,:,:,i]

    plt.subplot(6,6,i+1)
    plt.imshow(f)
    plt.title(f"{i}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("conv1_filters.png")
plt.show()