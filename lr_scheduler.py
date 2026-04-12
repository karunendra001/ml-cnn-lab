import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load CIFAR-10
# -----------------------------
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)

# 🔥 speed up
x_train = x_train[:10000]
y_train = y_train[:10000]

# -----------------------------
# Model (same as Problem 4 best → Both)
# -----------------------------
def build_model():
    model = models.Sequential([
        layers.Input(shape=(32,32,3)),

        layers.Conv2D(32,(3,3),padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),

        layers.Conv2D(64,(3,3),padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(10,activation='softmax')
    ])
    return model

# -----------------------------
# Cosine LR function
# -----------------------------
def cosine_lr(epoch):
    initial_lr = 0.001
    total_epochs = 30
    return initial_lr * (1 + np.cos(np.pi * epoch / total_epochs)) / 2

# -----------------------------
# Store results
# -----------------------------
histories = {}
lr_values = {}

# -----------------------------
# (a) ReduceLROnPlateau
# -----------------------------
model1 = build_model()
model1.compile(optimizer=tf.keras.optimizers.Adam(0.001),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

history1 = model1.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[reduce_lr],
    verbose=0
)

histories['ReduceLR'] = history1
lr_values['ReduceLR'] = [0.001]*30   # approximate

# -----------------------------
# (b) Cosine Annealing
# -----------------------------
model2 = build_model()
model2.compile(optimizer=tf.keras.optimizers.Adam(),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

lr_callback = tf.keras.callbacks.LearningRateScheduler(cosine_lr)

history2 = model2.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[lr_callback],
    verbose=0
)

histories['Cosine'] = history2
lr_values['Cosine'] = [cosine_lr(e) for e in range(30)]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,5))

# LR plot
plt.subplot(1,2,1)
for name in lr_values:
    plt.plot(lr_values[name], label=name)
plt.title("LR vs Epoch")
plt.legend()

# Accuracy plot
plt.subplot(1,2,2)
for name in histories:
    plt.plot(histories[name].history['val_accuracy'], label=name)
plt.title("Val Accuracy vs Epoch")
plt.legend()

plt.savefig("lr_schedule_comparison.png")
plt.show()