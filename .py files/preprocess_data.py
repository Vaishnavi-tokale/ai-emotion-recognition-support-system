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
images_dir = base_dir / "Dataset"              # folder containing emotion subfolders
labels_file = base_dir / "filtered_data.csv"   # your cleaned CSV file

# ------------------------------
# 🖼️ IMAGE CONFIGURATION
# ------------------------------
IMG_SIZE = (48, 48)

print("📂 Loading dataset and labels...")

# ------------------------------
# 🧾 READ LABEL CSV
# ------------------------------
df = pd.read_csv(labels_file)

# Ensure CSV has required columns
if 'path' not in df.columns or 'label' not in df.columns:
    raise ValueError("❌ CSV must contain columns: 'path' and 'label'")

X, y = [], []

# ------------------------------
# 🧠 LOAD & PREPROCESS IMAGES
# ------------------------------
for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = images_dir / row['path']
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    X.append(img)
    y.append(row['label'])

X = np.array(X)
y = np.array(y)

print(f"\n✅ Loaded {len(X)} images successfully.")
print("🖼️ Image array shape:", X.shape)
print("🎯 Unique labels:", np.unique(y))

# ------------------------------
# 🔁 DATA AUGMENTATION
# ------------------------------
print("\n🔄 Augmenting 'Happy' and 'Surprise' images to balance classes...")

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.6, 1.4],
    zoom_range=0.2,
    horizontal_flip=True
)

augmented_X, augmented_y = [], []
for img, label in zip(X, y):
    if label.lower() in ['happy', 'surprise']:  # augment underrepresented emotions
        img_expanded = np.expand_dims(img, 0)
        for _ in range(3):  # create 3 variants per image
            for batch in datagen.flow(img_expanded, batch_size=1):
                augmented_X.append(batch[0])
                augmented_y.append(label)
                break

if augmented_X:
    X = np.concatenate([X, np.array(augmented_X)])
    y = np.concatenate([y, np.array(augmented_y)])
    print(f"✨ Added {len(augmented_X)} augmented images.")
else:
    print("⚠️ No 'happy' or 'surprise' images found for augmentation.")

# ------------------------------
# 🔢 ENCODE LABELS
# ------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ------------------------------
# 💾 SAVE PROCESSED FILES
# ------------------------------
np.save(base_dir / "X.npy", X)
np.save(base_dir / "y.npy", y_encoded)
pd.DataFrame({'label': le.classes_, 'encoded': range(len(le.classes_))}).to_csv(base_dir / "label_map.csv", index=False)

print("\n💾 Saved files:")
print("   - X.npy (image data)")
print("   - y.npy (encoded labels)")
print("   - label_map.csv (emotion class mapping)")
print("✅ Preprocessing complete!")
