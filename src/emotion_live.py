import cv2
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model
from nlp_tts import respond_with_speech

# ---------------------------------
# LOAD MODEL
# ---------------------------------
model = load_model("trained_models/emotion_model_v1.h5")

# Label mapping from label_map.csv
emotion_labels = ['angry', 'happy', 'neutral', 'sad']

# ---------------------------------
# FACE DETECTOR
# ---------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------------
# SETTINGS
# ---------------------------------
emotion_buffer = deque(maxlen=5)

last_spoken_emotion = ""
last_spoken_time = 0

SPEAK_INTERVAL = 5

# ---------------------------------
# START CAMERA
# ---------------------------------
cap = cv2.VideoCapture(0)

print("📷 Camera Started")
print("Press 'Q' to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Face crop
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        # ---------------------------------
        # PREPROCESSING
        # ---------------------------------
        face = cv2.resize(face, (48, 48))

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=0)

        # ---------------------------------
        # PREDICTION
        # ---------------------------------
        prediction = model.predict(face, verbose=0)
        print(
    f"Angry:{prediction[0][0]:.2f} | "
    f"Happy:{prediction[0][1]:.2f} | "
    f"Neutral:{prediction[0][2]:.2f} | "
    f"Sad:{prediction[0][3]:.2f}"
)

        emotion_index = np.argmax(prediction)

        emotion = emotion_labels[emotion_index]

        confidence = np.max(prediction) * 100

        emotion_buffer.append(emotion)

        # Majority voting
        counts = {
            e: emotion_buffer.count(e)
            for e in set(emotion_buffer)
        }

        stable_emotion = max(counts, key=counts.get)

        # ---------------------------------
        # TEXT DISPLAY
        # ---------------------------------
        cv2.putText(
            frame,
            f"{stable_emotion} ({confidence:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ---------------------------------
        # VOICE OUTPUT
        # ---------------------------------
        current_time = time.time()

        if (
            stable_emotion != last_spoken_emotion
            and current_time - last_spoken_time > SPEAK_INTERVAL
        ):

            respond_with_speech(stable_emotion)

            last_spoken_emotion = stable_emotion

            last_spoken_time = current_time

    # ---------------------------------
    # WINDOW
    # ---------------------------------
    cv2.imshow(
        "AI Therapist - Emotion Detection",
        frame
    )

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

# ---------------------------------
# CLEANUP
# ---------------------------------
cap.release()
cv2.destroyAllWindows()

print("✅ Camera Closed")