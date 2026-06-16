import cv2
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model
from nlp_tts import respond_with_speech

model = load_model("Dataset/emotion_model.h5")

emotion_labels = ['angry', 'fear', 'happy', 'sad', 'neutral']

emotion_buffer = deque(maxlen=5)

last_spoken_time = 0
SPEAK_INTERVAL = 5   # speak every 5 seconds

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    resized = cv2.resize(frame, (48, 48))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=0)

    prediction = model.predict(reshaped, verbose=0)
    emotion = emotion_labels[np.argmax(prediction)]

    emotion_buffer.append(emotion)

    # Majority voting
    counts = {e: emotion_buffer.count(e) for e in set(emotion_buffer)}
    stable_emotion = max(counts, key=counts.get)

    current_time = time.time()

    # Speak every few seconds (even if same emotion)
    if current_time - last_spoken_time > SPEAK_INTERVAL:
        respond_with_speech(stable_emotion)
        last_spoken_time = current_time

    cv2.putText(frame, f"Stable Emotion: {stable_emotion}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    cv2.imshow("AI Therapist - Conversational Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
