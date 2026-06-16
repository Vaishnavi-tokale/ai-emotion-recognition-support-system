import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

DATA_PATH = "landmark_sequences"

X = []
y = []
label_map = {}

labels = sorted(os.listdir(DATA_PATH))

for index, label in enumerate(labels):
    label_map[index] = label
    label_path = os.path.join(DATA_PATH, label)

    for file in os.listdir(label_path):
        file_path = os.path.join(label_path, file)
        sequence = np.load(file_path)
        X.append(sequence)
        y.append(index)

X = np.array(X)
y = to_categorical(y)

print("Dataset shape:", X.shape)
print("Labels:", label_map)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(40,126)))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dense(len(labels), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test)
)

model.save("sign_model.h5")

with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Model trained and saved successfully!")