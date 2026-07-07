import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------
# 📂 PATH CONFIGURATION
# ------------------------------
base_dir = Path(r"E:\AI_Therapist_Project\Dataset")
images_dir = base_dir / "Dataset"
labels_file = base_dir / "filtered_data.csv"

# ------------------------------
# 🖼️ IMAGE CONFIGURATION
# ------------------------------
IMG_SIZE = (48, 48)

print("📂 Loading dataset and labels...")

# ------------------------------
# 🧾 READ LABEL CSV
# ------------------------------
df = pd.read_csv(labels_file)

# Keep only the 4 emotion classes
valid_emotions = ["Angry", "Happy", "Neutral", "Sad"]
df = df[df["label"].isin(valid_emotions)]

# Ensure CSV has required columns
if 'path' not in df.columns or 'label' not in df.columns:
    raise ValueError("❌ CSV must contain columns: 'path' and 'label'")

X = []
y = []

# ------------------------------
# 🧠 LOAD & PREPROCESS IMAGES
# ------------------------------
print("🖼️ Processing images...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = images_dir / row['path']

    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))

    if img is None:
        continue

    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0

    X.append(img)
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

print(f"\n✅ Loaded {len(X)} images successfully")
print("🖼️ Shape:", X.shape)
print("🎯 Labels:", np.unique(y))

# ------------------------------
# 🔄 DATA AUGMENTATION
# ------------------------------
print("\n🔄 Augmenting Angry images...")

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    brightness_range=[0.8, 1.2],
    zoom_range=0.15,
    horizontal_flip=True
)

augmented_X = []
augmented_y = []

for img, label in zip(X, y):

    # Only augment Angry class
    if label.lower() == "angry":

        img_expanded = np.expand_dims(img, axis=0)

        for _ in range(3):   # 3 augmented copies per image

            for batch in datagen.flow(img_expanded, batch_size=1):

                augmented_X.append(batch[0])
                augmented_y.append(label)
                break

if len(augmented_X) > 0:

    X = np.concatenate([X, np.array(augmented_X)])
    y = np.concatenate([y, np.array(augmented_y)])

    print(f"✨ Added {len(augmented_X)} augmented Angry images")

else:
    print("⚠️ No Angry images found for augmentation")

# ------------------------------
# 🔢 LABEL ENCODING
# ------------------------------
le = LabelEncoder()

y_encoded = le.fit_transform(y)

# ------------------------------
# 💾 SAVE FILES
# ------------------------------
np.save(base_dir / "X.npy", X)
np.save(base_dir / "y.npy", y_encoded)

label_map = pd.DataFrame({
    "label": le.classes_,
    "encoded": range(len(le.classes_))
})

label_map.to_csv(base_dir / "label_map.csv", index=False)

print("\n💾 Files saved:")
print("   X.npy")
print("   y.npy")
print("   label_map.csv")

print("\n📋 Label Mapping:")
print(label_map)

print("\n✅ Preprocessing Complete!")