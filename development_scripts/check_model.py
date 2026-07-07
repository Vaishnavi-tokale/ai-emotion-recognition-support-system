from tensorflow.keras.models import load_model
import os

MODEL_PATH = r"E:\AI_Therapist_Project\sign_lstm_model.h5"

print("Loading:", MODEL_PATH)
print("Exists:", os.path.exists(MODEL_PATH))

model = load_model(MODEL_PATH)

print("Output Shape:", model.output_shape)