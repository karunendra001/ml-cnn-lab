import tensorflow as tf

print("Loading CIFAR-10...")

# -----------------------------
# Load data
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Reduce dataset (memory safe)
x_train = x_train[:10000]
y_train = y_train[:10000]

# Resize
x_train = tf.image.resize(x_train, (96,96)).numpy()

# Preprocess
x_train = tf.keras.applications.vgg16.preprocess_input(x_train)

# -----------------------------
# Function to build model
# -----------------------------
def build_model():
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(96,96,3)
    )

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    return base_model, model

# -----------------------------
# Experiment function
# -----------------------------
def run_experiment(unfreeze_layers):
    base_model, model = build_model()

    # Freeze all layers
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze last N layers
    if unfreeze_layers == "all":
        for layer in base_model.layers:
            layer.trainable = True
    else:
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True

    # Count trainable params
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # EarlyStopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    # Train
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    best_val_acc = max(history.history['val_accuracy'])
    best_train_acc = max(history.history['accuracy'])

    # Overfitting check
    overfit = (best_train_acc - best_val_acc) > 0.05

    return trainable_params, best_val_acc, overfit


# -----------------------------
# Run experiments
# -----------------------------
results = []

configs = [2, 8, "all"]

for cfg in configs:
    print(f"\nRunning experiment: Unfreeze {cfg} layers")
    params, val_acc, overfit = run_experiment(cfg)

    results.append((cfg, params, val_acc, overfit))


# -----------------------------
# Print results table
# -----------------------------
print("\nAblation Study Results")
print("Layers | Trainable Params | Val Accuracy | Overfitting")

for r in results:
    print(f"{r[0]} | {r[1]} | {round(r[2],4)} | {r[3]}")