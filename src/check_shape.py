import os
import numpy as np

DATA_PATH = "landmark_sequences"

for label in os.listdir(DATA_PATH):
    label_path = os.path.join(DATA_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            file_path = os.path.join(label_path, file)
            data = np.load(file_path)
            print(label, file, "Shape:", data.shape)
            break