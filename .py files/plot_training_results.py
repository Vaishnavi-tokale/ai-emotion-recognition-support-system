import pandas as pd
import matplotlib.pyplot as plt
import os

# Path where training_history.csv is saved
base_path = r"E:\AI_Therapist_Project\Dataset"
history_path = os.path.join(base_path, "training_history_vgg19.csv")

# Load training history
history_df = pd.read_csv(history_path)
print("✅ Loaded training history successfully!")

# Plot training & validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history_df['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('📈 Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot training & validation loss
plt.figure(figsize=(8, 5))
plt.plot(history_df['loss'], label='Training Loss', color='red')
plt.plot(history_df['val_loss'], label='Validation Loss', color='green')
plt.title('📉 Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
