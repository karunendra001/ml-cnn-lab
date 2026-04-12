import tensorflow as tf
from tensorflow.keras import layers, models

# Build LeNet-5
model = models.Sequential([
    
    # C1
    layers.Conv2D(6, (5,5), activation='tanh', input_shape=(28,28,1)),
    
    # S2
    layers.AveragePooling2D((2,2)),
    
    # C3
    layers.Conv2D(16, (5,5), activation='tanh'),
    
    # S4
    layers.AveragePooling2D((2,2)),
    
    # Flatten
    layers.Flatten(),
    
    # C5
    layers.Dense(120, activation='tanh'),
    
    # F6
    layers.Dense(84, activation='tanh'),
    
    # Output
    layers.Dense(10, activation='softmax')
])

# Print summary
model.summary()