import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("model.h5")

# Build model
model(tf.zeros((1,32,32,3)))

# Load CIFAR-10
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

# Find last conv layer
for layer in model.layers[::-1]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

grad_model = tf.keras.Model(
    inputs=model.layers[0].input,
    outputs=[last_conv_layer.output, model.layers[-1].output]
)

# -----------------------------
# Get predictions
# -----------------------------
preds = model.predict(x_test[:200])
pred_labels = np.argmax(preds, axis=1)
true_labels = y_test[:200].flatten()

correct_idx = []
wrong_idx = []

for i in range(len(pred_labels)):
    if pred_labels[i] == true_labels[i] and len(correct_idx) < 3:
        correct_idx.append(i)
    elif pred_labels[i] != true_labels[i] and len(wrong_idx) < 1:
        wrong_idx.append(i)

indices = correct_idx + wrong_idx

# -----------------------------
# Grad-CAM function
# -----------------------------
def gradcam(img):
    img = np.expand_dims(img, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        grads = tf.ones_like(conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (32,32))
    return tf.squeeze(heatmap).numpy()

# -----------------------------
# Plot 4 images
# -----------------------------
plt.figure(figsize=(10,8))

for i, idx in enumerate(indices):
    img = x_test[idx]
    heatmap = gradcam(img)

    plt.subplot(2,2,i+1)
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(f"Pred:{pred_labels[idx]} True:{true_labels[idx]}")
    plt.axis('off')

plt.savefig("gradcam_multiple.png")
plt.show()