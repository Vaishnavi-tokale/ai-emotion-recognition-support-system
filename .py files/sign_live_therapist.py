import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import time

# ===============================
# SETTINGS
# ===============================
SEQUENCE_LENGTH = 40
CONFIDENCE_THRESHOLD = 0.80
COOLDOWN = 2   # seconds

# ===============================
# LOAD MODEL
# ===============================
model = load_model("sign_model.h5")

label_map = {
    0: "angry",
    1: "happy",
    2: "help",
    3: "no",
    4: "pain",
    5: "sad",
    6: "yes"
}

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
    "help": "I am here to help you. Please continue.",
    "pain": "I understand you are in pain. Please try to relax.",
    "sad": "It looks like you are feeling sad. I am here with you.",
    "angry": "Take a deep breath. Let us calm down together.",
    "happy": "I am glad you are feeling happy.",
    "yes": "Okay.",
    "no": "Alright."
}

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ===============================
# START CAMERA
# ===============================
cap = cv2.VideoCapture(0)

sequence = []
last_spoken_time = 0

print("AI Interactive Sign Therapist Started... Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame_landmarks = []

    # ===============================
    # EXTRACT LANDMARKS (ALWAYS 126)
    # ===============================
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])

    # Force 126 features
    if len(frame_landmarks) == 0:
        frame_landmarks = [0] * 126
    elif len(frame_landmarks) == 63:
        frame_landmarks.extend([0] * 63)
    elif len(frame_landmarks) > 126:
        frame_landmarks = frame_landmarks[:126]
    elif len(frame_landmarks) < 126:
        frame_landmarks.extend([0] * (126 - len(frame_landmarks)))

    sequence.append(frame_landmarks)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    # ===============================
    # PREDICTION
    # ===============================
    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(np.array(sequence), axis=0)
        prediction = model.predict(input_data, verbose=0)[0]

        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index]
        predicted_label = label_map[predicted_index]

        if confidence > CONFIDENCE_THRESHOLD:

            current_time = time.time()

            # 🔥 FIXED: allows repeated speech after cooldown
            if (current_time - last_spoken_time) > COOLDOWN:

                print(f"Detected: {predicted_label} | Confidence: {confidence:.2f}")

                response = responses.get(predicted_label, "I understand.")

                print("Therapist:", response)
                speak(response)

                print("--------------------------------")

                last_spoken_time = current_time

    cv2.imshow("AI Sign Therapist", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()