import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# 🔹 Path to dataset and model
base_path = r"E:\AI_Therapist_Project\Dataset"

print("📂 Loading test data and model...")
X_test = np.load(f"{base_path}\\X.npy")
y_test = np.load(f"{base_path}\\y.npy")

# ✅ Load your trained model
model = load_model(f"{base_path}\\emotion_model_vgg19.h5")

# If you already split train/test earlier, comment out next 3 lines
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42, stratify=y_test)

# Convert labels to categorical (for evaluation)
num_classes = len(set(y_test))
y_test_cat = to_categorical(y_test, num_classes)

# 🧪 Evaluate model
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"✅ Test Accuracy: {acc:.4f}")
print(f"🧮 Test Loss: {loss:.4f}")

# 🔍 Predict labels
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 📊 Classification report
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred_classes))

# 🔲 Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 🖼️ Show some predictions
emotion_map = pd.read_csv(f"{base_path}\\label_map.csv")
emotion_dict = dict(zip(emotion_map["Encoded"], emotion_map["Label"]))

plt.figure(figsize=(10,5))
for i in range(10):
    idx = np.random.randint(0, len(X_test))
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx])
    plt.title(f"Pred: {emotion_dict[y_pred_classes[idx]]}\nTrue: {emotion_dict[y_test[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
