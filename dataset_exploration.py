import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load datasets
# -----------------------------
mnist = tf.keras.datasets.mnist
cifar10 = tf.keras.datasets.cifar10

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()

# -----------------------------
# (a) Shapes
# -----------------------------
print("MNIST Train Shape:", x_train_mnist.shape)
print("MNIST Test Shape:", x_test_mnist.shape)

print("CIFAR-10 Train Shape:", x_train_cifar.shape)
print("CIFAR-10 Test Shape:", x_test_cifar.shape)

# -----------------------------
# (b) Data type & range
# -----------------------------
print("\nMNIST dtype:", x_train_mnist.dtype)
print("MNIST range:", x_train_mnist.min(), "to", x_train_mnist.max())

print("\nCIFAR dtype:", x_train_cifar.dtype)
print("CIFAR range:", x_train_cifar.min(), "to", x_train_cifar.max())

# -----------------------------
# (c) Class distribution (MNIST)
# -----------------------------
unique, counts = np.unique(y_train_mnist, return_counts=True)

print("\nMNIST Class Distribution:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

# -----------------------------
# Plot samples (2x10 grid)
# -----------------------------
fig, axes = plt.subplots(2, 10, figsize=(15, 4))

# MNIST samples
for i in range(10):
    idx = np.random.randint(0, len(x_train_mnist))
    axes[0, i].imshow(x_train_mnist[idx], cmap='gray')
    axes[0, i].set_title(y_train_mnist[idx])
    axes[0, i].axis('off')

# CIFAR class names
cifar_classes = ['airplane','automobile','bird','cat','deer',
                 'dog','frog','horse','ship','truck']

# CIFAR samples
for i in range(10):
    idx = np.random.randint(0, len(x_train_cifar))
    axes[1, i].imshow(x_train_cifar[idx])
    axes[1, i].set_title(cifar_classes[y_train_cifar[idx][0]])
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("dataset_samples.png")
plt.show()