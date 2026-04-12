import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -----------------------------
# Load MNIST
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# One-hot encode
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

# -----------------------------
# LeNet-5 Model
# -----------------------------
model = models.Sequential([
    layers.Conv2D(6,(5,5),activation='tanh',input_shape=(28,28,1)),
    layers.AveragePooling2D((2,2)),
    layers.Conv2D(16,(5,5),activation='tanh'),
    layers.AveragePooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(120,activation='tanh'),
    layers.Dense(84,activation='tanh'),
    layers.Dense(10,activation='softmax')
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1
)

# -----------------------------
# Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# -----------------------------
# Plot Loss
# -----------------------------
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Loss Curve")

# Mark overfitting point manually later
plt.savefig("lenet_sgd_loss.png")

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title("Accuracy Curve")

plt.savefig("lenet_sgd_accuracy.png")

plt.show()