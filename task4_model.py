import tensorflow as tf
from tensorflow.keras import layers, models

# Load CIFAR-10
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

# Normalize
x_train = x_train.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)

# Small data for speed
x_train = x_train[:10000]
y_train = y_train[:10000]

# Model
model = models.Sequential([
    layers.Input(shape=(32,32,3)),

    layers.Conv2D(32,(3,3),padding='same',activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=3, batch_size=64)

# Save model
model.save("model.h5")

print("Model saved")