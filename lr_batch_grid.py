import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# Load CIFAR-10
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# -----------------------------
# Model (same as Task 2)
# -----------------------------
def build_model():
    model = models.Sequential([
        layers.Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64,(3,3),padding='same',activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])
    return model

# -----------------------------
# Hyperparameters
# -----------------------------
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [32, 128]

results = {}

# -----------------------------
# Grid Search
# -----------------------------
for lr in learning_rates:
    for bs in batch_sizes:
        print(f"\nTraining with LR={lr}, Batch Size={bs}")

        model = build_model()  # IMPORTANT: fresh model

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            x_train, y_train,
            epochs=10,
            batch_size=bs,
            validation_split=0.1,
            verbose=0
        )

        val_acc = history.history['val_accuracy'][-1]
        results[(lr, bs)] = val_acc

# -----------------------------
# Print Table
# -----------------------------
print("\nFinal Validation Accuracy Table:\n")

for lr in learning_rates:
    row = []
    for bs in batch_sizes:
        row.append(f"{results[(lr, bs)]:.4f}")
    print(f"LR={lr}: {row}")