import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam

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
# Freeze all layers except the last 8 layers
for layer in base_model.layers[:-8]:
    layer.trainable = False
for layer in base_model.layers[-8:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
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

# Calculate class weights dynamically
y_train_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_integers),
    y=y_train_integers
)
class_weight_dict = dict(zip(np.unique(y_train_integers), class_weights))

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=35,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weight_dict,
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
