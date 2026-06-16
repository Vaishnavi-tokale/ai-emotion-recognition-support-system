import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ------------------------------
# 📂 PATH CONFIGURATION
# ------------------------------
base_path = r"E:\AI_Therapist_Project\Dataset"
X_path = base_path + "\\X.npy"
y_path = base_path + "\\y.npy"

print("📂 Loading preprocessed data...")
X = np.load(X_path)
y = np.load(y_path)

# ------------------------------
# 🧩 DATA PREPARATION
# ------------------------------
num_classes = len(set(y))
y_cat = to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data split complete! Train: {X_train.shape}, Test: {X_test.shape}")

# ------------------------------
# 🧠 MODEL DEFINITION (VGG19)
# ------------------------------
print("🔧 Building VGG19-based CNN model...")

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
for layer in base_model.layers:
    layer.trainable = False  # freeze pretrained layers

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# 🪄 CALLBACKS
# ------------------------------
checkpoint_path = base_path + "\\emotion_model_vgg19.h5"

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy')
]

# ------------------------------
# 🚀 TRAINING
# ------------------------------
print("🚀 Training started...")

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=25,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------
# 📊 SAVE TRAINING HISTORY
# ------------------------------
history_df = pd.DataFrame(history.history)
history_df.to_csv(base_path + "\\training_history_vgg19.csv", index=False)
print("💾 Training history saved to training_history_vgg19.csv")

# ------------------------------
# 🧾 EVALUATION
# ------------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test accuracy: {acc:.4f}")
print(f"📉 Test loss: {loss:.4f}")

print(f"💾 Model saved to: {checkpoint_path}")
