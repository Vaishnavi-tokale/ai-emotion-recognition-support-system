import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import time

# ===============================
# SETTINGS
# ===============================
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.70
COOLDOWN = 2

# ===============================
# LOAD MODEL
# ===============================
model = load_model("sign_lstm_model.h5")

print("Model Output Shape:", model.output_shape)

import pickle

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# ===============================
# TEXT TO SPEECH
# ===============================
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ===============================
# RESPONSES
# ===============================
responses = {
    "help": "I can see that you are asking for help. I am here to support you.",
    "yes": "Thank you for confirming. Let's continue.",
    "no": "That's okay. We can try another approach."
}

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

sequence = []
last_spoken_time = 0

print("AI Interactive Sign Therapist Started...")
print("Press Q to quit")

# ===============================
# MAIN LOOP
# ===============================
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    image = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    frame_landmarks = []

    # ===============================
    # EXTRACT LANDMARKS (63 FEATURES)
    # ===============================
    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        for lm in hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z])

    # Force exactly 63 features
    if len(frame_landmarks) == 0:
        frame_landmarks = [0] * 63

    elif len(frame_landmarks) > 63:
        frame_landmarks = frame_landmarks[:63]

    elif len(frame_landmarks) < 63:
        frame_landmarks.extend([0] * (63 - len(frame_landmarks)))

    sequence.append(frame_landmarks)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    # ===============================
    # PREDICTION
    # ===============================
    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.expand_dims(
            np.array(sequence),
            axis=0
        )

        prediction = model.predict(
            input_data,
            verbose=0
        )[0]

        print("Raw Prediction:", prediction)

        predicted_index = int(np.argmax(prediction))

        print("Predicted Index:", predicted_index)

        if predicted_index not in label_map:
            print("Invalid prediction index:", predicted_index)
            continue

        confidence = float(
            prediction[predicted_index]
        )

        predicted_label = label_map.get(predicted_index, "Unknown")
        cv2.putText(
    image,
    f"{predicted_label} ({confidence:.2f})",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

        cv2.putText(
            image,
            f"{predicted_label} ({confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        if confidence > CONFIDENCE_THRESHOLD:

            current_time = time.time()

            if (
                current_time - last_spoken_time
            ) > COOLDOWN:

                print(
                    f"Detected: {predicted_label} | Confidence: {confidence:.2f}"
                )

                response = responses.get(
                    predicted_label,
                    "I understand."
                )

                print("Therapist:", response)

                speak(response)

                print("--------------------------------")

                last_spoken_time = current_time

    cv2.imshow(
        "AI Sign Therapist",
        image
    )

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()

print("Camera Closed")