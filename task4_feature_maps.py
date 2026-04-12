import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = tf.keras.models.load_model("model.h5")

# Load CIFAR-10
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

# Select image
img = x_test[0]
img = np.expand_dims(img, axis=0)

# Get Conv layers
conv_layers = []
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        conv_layers.append(layer.output)

# 🔥 FIXED LINE HERE
feature_model = tf.keras.Model(inputs=model.inputs, outputs=conv_layers)

# Predict
feature_maps = feature_model.predict(img)

# First layer
fmap1 = feature_maps[0][0]

plt.figure(figsize=(8,8))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(fmap1[:,:,i], cmap='gray')
    plt.axis('off')

plt.savefig("fmaps_layer1.png")
plt.show()

# Last layer
fmap_last = feature_maps[-1][0]

plt.figure(figsize=(8,8))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(fmap_last[:,:,i], cmap='gray')
    plt.axis('off')

plt.savefig("fmaps_last.png")
plt.show()