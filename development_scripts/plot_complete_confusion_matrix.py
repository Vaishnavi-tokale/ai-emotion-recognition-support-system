import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import os

base_path = r"E:\AI_Therapist_Project\Dataset"
X_path = os.path.join(base_path, "X.npy")
y_path = os.path.join(base_path, "y.npy")
model_path = os.path.join(base_path, "emotion_model_vgg19.h5")

print("📂 Loading the complete dataset...")
try:
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"✅ Loaded X shape: {X.shape}, y shape: {y.shape}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

print("🔧 Loading the trained VGG19 model...")
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

print("🔮 Generating predictions on the complete dataset...")
y_pred = model.predict(X)
y_pred_labels = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y, y_pred_labels)
print(f"\n✅ Accuracy on the Complete Dataset: {accuracy * 100:.2f}%")

cm = confusion_matrix(y, y_pred_labels)

# Assuming these are the 5 classes standard to the project based on other scripts
labels = ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprise']

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)

plt.title(f'Confusion Matrix (Complete Training Data)\nAccuracy: {accuracy*100:.2f}%')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

save_path = os.path.join(base_path, "confusion_matrix_complete.png")
plt.savefig(save_path, dpi=300)
print(f"💾 Confusion matrix saved successfully to: {save_path}")

try:
    plt.show()
except:
    pass
print("✅ Done!")
