import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

DATA_PATH = "landmark_sequences"
SEQUENCE_LENGTH = 30

X = []
y = []

labels = sorted(os.listdir(DATA_PATH))

for label in labels:
    label_path = os.path.join(DATA_PATH, label)

    for file in os.listdir(label_path):
        sequence = np.load(os.path.join(label_path, file))

        sequence = pad_sequences(
            [sequence],
            maxlen=SEQUENCE_LENGTH,
            padding='post'
        )[0]

        X.append(sequence)
        y.append(label)

X = np.array(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

np.save("X.npy", X)
np.save("y.npy", y_categorical)
np.save("label_classes.npy", le.classes_)

print("Data prepared successfully.")
print("Shape of X:", X.shape)
print("Number of classes:", len(le.classes_))
