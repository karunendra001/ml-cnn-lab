import tensorflow as tf
import matplotlib.pyplot as plt

print("Step 1: Load CIFAR-10")

# -----------------------------
# Load dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# -----------------------------
# Reduce dataset (IMPORTANT)
# -----------------------------
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]

# -----------------------------
# Resize (memory safe)
# -----------------------------
x_train = tf.image.resize(x_train, (96,96)).numpy()
x_test = tf.image.resize(x_test, (96,96)).numpy()

# -----------------------------
# Preprocess for VGG16
# -----------------------------
x_train = tf.keras.applications.vgg16.preprocess_input(x_train)
x_test = tf.keras.applications.vgg16.preprocess_input(x_test)

print("Step 2: Load VGG16")

# -----------------------------
# Load pretrained VGG16
# -----------------------------
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)

# -----------------------------
# Freeze all layers (Problem 1)
# -----------------------------
for layer in base_model.layers:
    layer.trainable = False

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
# Compile (Problem 1)
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Step 3: Train Frozen Model")

# -----------------------------
# Train frozen model (epochs 1–10)
# -----------------------------
history1 = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# ======================================================
#  FINE-TUNING PART
# ======================================================

print("Step 4: Unfreeze last 4 layers")

# -----------------------------
# Unfreeze last 4 layers
# -----------------------------
for layer in base_model.layers[-4:]:
    layer.trainable = True

# -----------------------------
# IMPORTANT: Recompile
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Step 5: Fine-Tuning")

# -----------------------------
# EarlyStopping
# -----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# Train (epochs 11–20)
# -----------------------------
history2 = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop]
)

# -----------------------------
# Combine results
# -----------------------------
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

# -----------------------------
# Plot combined graph
# -----------------------------
plt.figure()

plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')

# Mark transition (epoch 10 → 11)
plt.axvline(x=9, linestyle='--', label='Fine-tuning Start')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Transfer Learning + Fine-Tuning (VGG16)")
plt.legend()

plt.savefig("tl_finetuned.png")
plt.show()

print("Graph saved as tl_finetuned.png")

# -----------------------------
# Best epoch
# -----------------------------
best_epoch = val_acc.index(max(val_acc)) + 1
best_acc = max(val_acc)

print("\nBest Epoch:", best_epoch)
print("Best Validation Accuracy:", best_acc)