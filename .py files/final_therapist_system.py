import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import time
import pickle
from collections import deque

# =========================
# LOAD MODELS
# =========================

emotion_model = load_model("model.hdf5", compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

sign_model = load_model("sign_lstm_model.h5")

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

label_map = {v: k for k, v in label_map.items()}

# =========================
# FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# =========================
# HAND DETECTOR
# =========================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# =========================
# VOICE ENGINE
# =========================

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# =========================
# THERAPY RESPONSE
# =========================

def therapy_response(emotion):
    responses = {
        "Happy": "It's great to see you happy!",
        "Sad": "I understand. Take a deep breath.",
        "Angry": "Try to relax. Breathe slowly.",
        "Fear": "You are safe. Stay calm.",
        "Distress": "I can see you need help. I am here for you.",
        "Need Help": "Do you need assistance? I am listening."
    }
    return responses.get(emotion, "I'm here with you.")

# =========================
# SIGN BUFFER
# =========================

sequence = []
SEQUENCE_LENGTH = 30

# =========================
# EMOTION SMOOTHING
# =========================

emotion_buffer = deque(maxlen=10)

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

last_spoken_time = 0
SPEAK_DELAY = 4

# =========================
# TEXT DRAW FUNCTION (NO BACKGROUND)
# =========================

def draw_text(img, text, pos, color):
    x, y = pos
    # shadow
    cv2.putText(img, text, (x+2, y+2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
    # main text
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# =========================
# MAIN LOOP
# =========================

while True:
    try:
        ret, frame = cap.read()

        if not ret:
            print("Camera read failed... retrying")
            continue

        frame = cv2.flip(frame, 1)

        # =========================
        # FACE EMOTION
        # =========================

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_emotion = "Neutral"

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            face = np.reshape(face, (1, 64, 64, 1))

            prediction = emotion_model.predict(face, verbose=0)
            confidence = np.max(prediction)

            if confidence < 0.6:
                continue

            raw_emotion = emotion_labels[np.argmax(prediction)]

            emotion_buffer.append(raw_emotion)
            detected_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # =========================
        # HAND + SIGN DETECTION
        # =========================

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        keypoints = []
        predicted_sign = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

        if len(keypoints) == 0:
            keypoints = [0]*63

        sequence.append(keypoints)

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        if len(sequence) == SEQUENCE_LENGTH:
            try:
                input_data = np.expand_dims(sequence, axis=0)
                prediction = sign_model.predict(input_data, verbose=0)
                predicted_class = np.argmax(prediction)
                predicted_sign = label_map.get(predicted_class, None)
            except Exception as e:
                print("Sign model error:", e)
                predicted_sign = None

        # =========================
        # FUSION LOGIC
        # =========================

        final_emotion = detected_emotion

        if detected_emotion in ["Sad", "Fear"]:
            final_emotion = "Sad"

        if predicted_sign:
            if predicted_sign in ["help", "please"]:
                final_emotion = "Distress"
            elif predicted_sign in ["yes", "good"]:
                final_emotion = "Happy"
            elif predicted_sign in ["no", "stop"]:
                final_emotion = "Angry"

        # =========================
        # RESPONSE
        # =========================

        response = therapy_response(final_emotion)

        current_time = time.time()
        if current_time - last_spoken_time > SPEAK_DELAY:
            speak(response)
            last_spoken_time = current_time

        # =========================
        # DISPLAY (NO BLACK BOX)
        # =========================

        draw_text(frame, f"Emotion: {final_emotion}", (10, 40), (0,255,255))

        if predicted_sign:
            draw_text(frame, f"Sign: {predicted_sign}", (10, 80), (255,255,0))

        draw_text(frame, f"Response: {response}", (10, 120), (0,255,0))

        cv2.imshow("AI Therapist System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Runtime error:", e)
        continue

cap.release()
cv2.destroyAllWindows()