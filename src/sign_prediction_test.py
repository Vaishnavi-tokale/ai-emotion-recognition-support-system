import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model
model = load_model("sign_lstm_model.h5")
labels = np.load("label_classes.npy")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sequence = []

cap = cv2.VideoCapture(0)

print("Prediction test started...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            sequence.append(landmarks)

            # Keep last 30 frames
            if len(sequence) > 30:
                sequence.pop(0)

            print("Sequence length:", len(sequence))

            if len(sequence) == 30:
                input_data = np.expand_dims(sequence, axis=0)
                prediction = model.predict(input_data, verbose=0)[0]

                predicted_class = labels[np.argmax(prediction)]
                confidence = np.max(prediction)

                print("Predicted:", predicted_class)
                print("Confidence:", confidence)
                print("-------------")

    cv2.imshow("Prediction Test", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
