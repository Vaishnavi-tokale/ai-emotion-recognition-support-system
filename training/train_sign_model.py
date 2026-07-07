import os
import numpy as np
import pickle
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# ==================================
# CONFIGURATION
# ==================================
DATA_PATH = "landmark_sequences"

SEQUENCE_LENGTH = 30
FEATURES = 63

# ==================================
# LOAD DATA
# ==================================
X = []
y = []
label_map = {}

print("📂 Loading gesture sequences...")

labels = sorted(os.listdir(DATA_PATH))

for index, label in enumerate(labels):

    label_map[index] = label

    label_path = os.path.join(DATA_PATH, label)

    for file in os.listdir(label_path):

        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(label_path, file)

        sequence = np.load(file_path)

        # Skip invalid sequences
        if sequence.shape != (SEQUENCE_LENGTH, FEATURES):
            print(
                f"⚠ Skipping {file} | Shape: {sequence.shape}"
            )
            continue

        X.append(sequence)
        y.append(index)

# ==================================
# CONVERT TO NUMPY
# ==================================
X = np.array(X)
y = np.array(y)

print(f"\n✅ Dataset Loaded")
print(f"📊 X Shape: {X.shape}")
print(f"📊 y Shape: {y.shape}")
print(f"🏷 Labels: {label_map}")

# ==================================
# ONE-HOT ENCODING
# ==================================
y_cat = to_categorical(y, num_classes=len(labels))

# ==================================
# TRAIN TEST SPLIT
# ==================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n✅ Train Shape: {X_train.shape}")
print(f"✅ Test Shape: {X_test.shape}")

# ==================================
# BUILD MODEL
# ==================================
print("\n🧠 Building LSTM Model...")

model = Sequential()

model.add(
    LSTM(
        128,
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, FEATURES)
    )
)

model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(LSTM(128))

model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dense(len(labels), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==================================
# EARLY STOPPING
# ==================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
# ==================================
# MODEL CHECKPOINT
# ==================================
checkpoint = ModelCheckpoint(
    "best_sign_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
# ==================================
# TRAIN MODEL
# ==================================
print("\n🚀 Training Started...")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ==================================
# EVALUATION
# ==================================
print("\n📊 Evaluating Model...")

loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"\n✅ Gesture Test Accuracy: {acc:.4f}")
print(f"📉 Gesture Test Loss: {loss:.4f}")
# ==================================
# SAVE TRAINING HISTORY
# ==================================
history_df = pd.DataFrame(history.history)
history_df.to_csv("gesture_training_history.csv", index=False)

print("💾 Training history saved as gesture_training_history.csv")

# ==================================
# SAVE MODEL
# ==================================
model.save("sign_lstm_model.h5")

with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("\n💾 Model saved as sign_lstm_model.h5")
print("💾 Label map saved as label_map.pkl")

print("\n🎉 Training Completed Successfully!")