import tensorflow as tf
import matplotlib.pyplot as plt

print("Loading CIFAR-10...")

# -----------------------------
# Load data
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Reduce dataset
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]

# Normalize for scratch model
x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0

# Resize for VGG
x_train_vgg = tf.image.resize(x_train, (96,96)).numpy()
x_test_vgg = tf.image.resize(x_test, (96,96)).numpy()

x_train_vgg = tf.keras.applications.vgg16.preprocess_input(x_train_vgg)
x_test_vgg = tf.keras.applications.vgg16.preprocess_input(x_test_vgg)

# =========================================================
# 1️⃣ SCRATCH MODEL
# =========================================================
def build_scratch():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

scratch_model = build_scratch()

scratch_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Scratch Model")

hist_scratch = scratch_model.fit(
    x_train_norm, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

test_acc_scratch = scratch_model.evaluate(x_test_norm, y_test, verbose=0)[1]

# =========================================================
# 2️⃣ FROZEN VGG16
# =========================================================
def build_vgg_frozen():
    base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(96,96,3))
    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=base.input, outputs=output)
    return model

vgg_frozen = build_vgg_frozen()

vgg_frozen.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Frozen VGG16")

hist_frozen = vgg_frozen.fit(
    x_train_vgg, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

test_acc_frozen = vgg_frozen.evaluate(x_test_vgg, y_test, verbose=0)[1]

# =========================================================
# 3️⃣ FINE-TUNED VGG16
# =========================================================
def build_vgg_finetune():
    base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(96,96,3))

    # Freeze first
    for layer in base.layers:
        layer.trainable = False

    # Unfreeze last 8 layers (best)
    for layer in base.layers[-8:]:
        layer.trainable = True

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=base.input, outputs=output)
    return model

vgg_ft = build_vgg_finetune()

vgg_ft.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Fine-Tuned VGG16")

hist_ft = vgg_ft.fit(
    x_train_vgg, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

test_acc_ft = vgg_ft.evaluate(x_test_vgg, y_test, verbose=0)[1]

# =========================================================
# 📊 PLOT COMPARISON
# =========================================================
plt.figure()

plt.plot(hist_scratch.history['val_accuracy'], label='Scratch CNN')
plt.plot(hist_frozen.history['val_accuracy'], label='Frozen VGG16')
plt.plot(hist_ft.history['val_accuracy'], label='Fine-tuned VGG16')

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Benchmark Comparison")
plt.legend()

plt.savefig("tl_benchmark.png")
plt.show()

# =========================================================
# 📋 PRINT TABLE
# =========================================================
print("\nBenchmark Results")
print("Model | Test Accuracy")

print(f"Scratch CNN | {round(test_acc_scratch,4)}")
print(f"Frozen VGG16 | {round(test_acc_frozen,4)}")
print(f"Fine-tuned VGG16 | {round(test_acc_ft,4)}")