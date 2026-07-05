import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import os

base_path = "Dataset"
model_path = os.path.join(base_path, "emotion_model_vgg19.h5")

# Load model
print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)

# Load test data
print("Loading y_test.npy and test_idx.npy...")
y_test = np.load(os.path.join(base_path, "y_test.npy"))
X = np.load(os.path.join(base_path, "X.npy"))
test_idx = np.load(os.path.join(base_path, "test_idx.npy"))

# Subsample X to get X_test
X_test = X[test_idx]

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Evaluate
print("Predicting on test set...")
preds = model.predict(X_test, verbose=0)
pred_classes = np.argmax(preds, axis=1)

# Map y_test if it is one-hot
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    true_classes = np.argmax(y_test, axis=1)
else:
    true_classes = y_test

label_map = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprise"}

print("\n--- Prediction Frequency on Test Set ---")
unique, counts = np.unique(pred_classes, return_counts=True)
total = len(pred_classes)
pred_dist = {u: c for u, c in zip(unique, counts)}
for i in range(5):
    c = pred_dist.get(i, 0)
    label = label_map.get(i)
    print(f"  Predicted {label} ({i}): {c} times ({c/total*100:.2f}%)")

print("\n--- Actual Class Frequency on Test Set ---")
unique_true, counts_true = np.unique(true_classes, return_counts=True)
total_true = len(true_classes)
true_dist = {u: c for u, c in zip(unique_true, counts_true)}
for i in range(5):
    c = true_dist.get(i, 0)
    label = label_map.get(i)
    print(f"  Actual {label} ({i}): {c} times ({c/total_true*100:.2f}%)")

print("\n--- Classification Report ---")
target_names = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
print(classification_report(true_classes, pred_classes, target_names=target_names))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(true_classes, pred_classes)
print("Row=Actual, Col=Predicted")
print(f"         {target_names}")
for idx, row in enumerate(cm):
    print(f"{target_names[idx]:8}: {row}")
