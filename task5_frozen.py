import tensorflow as tf
import matplotlib.pyplot as plt

print("Step 1: Loading CIFAR-10...")

# -----------------------------
# Load dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# -----------------------------
# Reduce dataset (VERY IMPORTANT)
# -----------------------------
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]

print("Step 2: Resizing images...")

# -----------------------------
# Resize images (safe)
# -----------------------------
x_train = tf.image.resize(x_train, (96, 96)).numpy()
x_test = tf.image.resize(x_test, (96, 96)).numpy()

print("Step 3: Preprocessing...")

# -----------------------------
# Preprocess for VGG16
# -----------------------------
x_train = tf.keras.applications.vgg16.preprocess_input(x_train)
x_test = tf.keras.applications.vgg16.preprocess_input(x_test)

print("Step 4: Loading VGG16...")

# -----------------------------
# Load pretrained VGG16
# -----------------------------
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(96, 96, 3)
)

# -----------------------------
# Freeze all layers
# -----------------------------
for layer in base_model.layers:
    layer.trainable = False

print("Step 5: Building model...")

# -----------------------------
# Add custom head
# -----------------------------
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# -----------------------------
# Compile model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

print("Step 6: Training...")

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

print("Step 7: Plotting...")

# -----------------------------
# Plot graph
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Transfer Learning - Frozen VGG16")
plt.legend()

plt.savefig("tl_frozen.png")
plt.show()

print("DONE ")