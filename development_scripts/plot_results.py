import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# ===========================
# LOAD TRAINING HISTORY
# ===========================

history = pd.read_csv(r"Dataset/training_history_vgg19.csv")

# ===========================
# ACCURACY CURVE
# ===========================

plt.figure(figsize=(8,5))

plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("accuracy_curve.png", dpi=300)
plt.show()

# ===========================
# LOSS CURVE
# ===========================

plt.figure(figsize=(8,5))

plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("loss_curve.png", dpi=300)
plt.show()

# ===========================
# CONFUSION MATRIX
# ===========================

print("Loading model...")

model = load_model(r"Dataset/emotion_model_vgg19.h5")

X = np.load(r"Dataset/X.npy")
y = np.load(r"Dataset/y.npy")

num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pred = model.predict(X_test)

y_pred = np.argmax(pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

labels = ["Angry","Happy","Neutral","Sad"]

plt.figure(figsize=(7,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("Graphs Saved Successfully.")