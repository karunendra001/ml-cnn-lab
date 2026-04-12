import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model("model.h5")

# -----------------------------
# Load CIFAR-10
# -----------------------------
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0
y_test = y_test.flatten()

# Class names
class_names = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# -----------------------------
# Predictions
# -----------------------------
preds = model.predict(x_test)
y_pred = np.argmax(preds, axis=1)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()

# -----------------------------
# Classification Report
# -----------------------------
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

# -----------------------------
# Find best & worst F1
# -----------------------------
from sklearn.metrics import precision_recall_fscore_support

_, _, f1, _ = precision_recall_fscore_support(y_test, y_pred)

best_class = class_names[np.argmax(f1)]
worst_class = class_names[np.argmin(f1)]

print("\nBest F1-score class:", best_class)
print("Worst F1-score class:", worst_class)

# -----------------------------
# Most confused pair
# -----------------------------
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

i, j = np.unravel_index(np.argmax(cm_no_diag), cm.shape)

print("\nMost confused pair:", class_names[i], "->", class_names[j])

# -----------------------------
# Show 5 misclassified examples
# -----------------------------
indices = np.where((y_test == i) & (y_pred == j))[0][:5]

plt.figure(figsize=(10,5))
for k, idx in enumerate(indices):
    plt.subplot(1,5,k+1)
    plt.imshow(x_test[idx])
    plt.title(f"T:{class_names[i]}\nP:{class_names[j]}")
    plt.axis('off')

plt.savefig("confused_examples.png")
plt.show()