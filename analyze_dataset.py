import numpy as np
import pandas as pd
import os

base_path = "Dataset"
y_path = os.path.join(base_path, "y.npy")
y_train_path = os.path.join(base_path, "y_train.npy")
y_test_path = os.path.join(base_path, "y_test.npy")

label_map = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprise"}

print("--- Dataset Frequencies ---")
if os.path.exists(y_path):
    y = np.load(y_path)
    unique, counts = np.unique(y, return_counts=True)
    print("Full Dataset Distribution (y.npy):")
    total = len(y)
    for u, c in zip(unique, counts):
        label = label_map.get(u, f"Unknown ({u})")
        print(f"  {label} ({u}): {c} images ({c/total*100:.2f}%)")
else:
    print("y.npy not found")

if os.path.exists(y_train_path):
    y_train = np.load(y_train_path)
    # Check if y_train is one-hot encoded or label encoded
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train_labels = np.argmax(y_train, axis=1)
    else:
        y_train_labels = y_train
    unique, counts = np.unique(y_train_labels, return_counts=True)
    print("\nTrain Dataset Distribution (y_train.npy):")
    total = len(y_train_labels)
    for u, c in zip(unique, counts):
        label = label_map.get(u, f"Unknown ({u})")
        print(f"  {label} ({u}): {c} images ({c/total*100:.2f}%)")

if os.path.exists(y_test_path):
    y_test = np.load(y_test_path)
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test
    unique, counts = np.unique(y_test_labels, return_counts=True)
    print("\nTest Dataset Distribution (y_test.npy):")
    total = len(y_test_labels)
    for u, c in zip(unique, counts):
        label = label_map.get(u, f"Unknown ({u})")
        print(f"  {label} ({u}): {c} images ({c/total*100:.2f}%)")
