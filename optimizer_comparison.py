import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -----------------------------
# Load MNIST
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocessing
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# -----------------------------
# Model function (LeNet-5)
# -----------------------------
def build_model():
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),

        layers.Conv2D(6, (5,5), activation='tanh'),
        layers.AveragePooling2D((2,2)),

        layers.Conv2D(16, (5,5), activation='tanh'),
        layers.AveragePooling2D((2,2)),

        layers.Flatten(),

        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),

        layers.Dense(10, activation='softmax')
    ])
    return model

histories = {}

# -----------------------------
# (a) SGD
# -----------------------------
model_sgd = build_model()
model_sgd.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

histories['SGD'] = model_sgd.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)

# -----------------------------
# (b) SGD + Momentum
# -----------------------------
model_momentum = build_model()
model_momentum.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

histories['SGD + Momentum'] = model_momentum.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)

# -----------------------------
# (c) Adam
# -----------------------------
model_adam = build_model()
model_adam.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

histories['Adam'] = model_adam.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)

# -----------------------------
# Plot Validation Accuracy
# -----------------------------
plt.figure()

for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=name)

plt.title("Optimizer Comparison (Validation Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()

plt.savefig("optimiser_comparison.png")
plt.show()