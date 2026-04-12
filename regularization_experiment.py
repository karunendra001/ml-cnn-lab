import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)

# -----------------------------
# Model builder
# -----------------------------
def build_model(use_dropout=False, use_bn=False):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32,(3,3),padding='same',input_shape=(32,32,3)))
    if use_bn:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    if use_dropout:
        model.add(layers.Dropout(0.3))

    # Block 2
    model.add(layers.Conv2D(64,(3,3),padding='same'))
    if use_bn:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    if use_dropout:
        model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    if use_dropout:
        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10, activation='softmax'))

    return model

# -----------------------------
# Variants
# -----------------------------
configs = {
    "No Reg": (False, False),
    "Dropout": (True, False),
    "BatchNorm": (False, True),
    "Both": (True, True)
}

results = {}

# -----------------------------
# Train all
# -----------------------------
for name, (drop, bn) in configs.items():
    print(f"\nTraining: {name}")

    model = build_model(use_dropout=drop, use_bn=bn)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )

    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    gap = train_acc - val_acc

    results[name] = gap

    # Plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title(name)
    plt.legend()
    plt.savefig(f"{name}_curve.png")

# -----------------------------
# Print results
# -----------------------------
print("\nTrain-Val Accuracy Gaps:\n")
for k, v in results.items():
    print(f"{k}: {v:.4f}")