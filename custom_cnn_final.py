import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([

    # Block 1
    layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    # Block 2
    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    # Block 3
    layers.Conv2D(128, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    # Global Pooling
    layers.GlobalAveragePooling2D(),

    # Dense Head
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    # Output
    layers.Dense(10, activation='softmax')
])

model.summary()