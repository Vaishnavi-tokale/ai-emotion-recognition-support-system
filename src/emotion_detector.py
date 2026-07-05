import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model


class EmotionDetector:

    def __init__(self, model=None):

        # -----------------------------
        # Load Emotion Model
        # -----------------------------
        if model is not None:
            self.model = model
        else:
            self.model = load_model(
                "trained_models/emotion_model_v1.h5"
            )

        # Emotion Labels
        self.emotion_labels = [
            "angry",
            "happy",
            "neutral",
            "sad"
        ]

        # Face Detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        # Majority Voting Buffer
        self.buffer = deque(maxlen=5)

        print("[OK] Emotion Detector Loaded")

    # -------------------------------------------------
    # Detect Emotion
    # -------------------------------------------------
    def detect(self, frame, predict=True):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Performance optimization: downscale grayscale image by 2x for faster face detection
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

        faces = self.face_detector.detectMultiScale(
            small_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)  # Halved minSize because resolution is halved
        )

        detected_emotion = "Unknown"
        confidence = 0

        for (x_small, y_small, w_small, h_small) in faces:
            # Scale coordinates back to original size
            x, y, w, h = x_small * 2, y_small * 2, w_small * 2, h_small * 2

            # Draw Rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            if not predict:
                if len(self.buffer) > 0:
                    counts = {e: self.buffer.count(e) for e in set(self.buffer)}
                    detected_emotion = max(counts, key=counts.get)
                    cv2.putText(
                        frame,
                        f"{detected_emotion}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                continue

            # Add padding to face crop to match standard emotion dataset cropping (like FER2013)
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)
            
            y1 = max(0, y - pad_h)
            y2 = min(frame.shape[0], y + h + pad_h)
            x1 = max(0, x - pad_w)
            x2 = min(frame.shape[1], x + w + pad_w)
            
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # -----------------------------
            # Preprocessing
            # -----------------------------
            face = cv2.resize(
                face,
                (48, 48)
            )

            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            face = face.astype("float32") / 255.0

            face = np.expand_dims(
                face,
                axis=0
            )

            # -----------------------------
            # Prediction
            # -----------------------------
            prediction = self.model(
                face,
                training=False
            ).numpy()

            pred_probs = prediction[0].copy()
            # Calibration weights to balance live camera predictions
            pred_probs[3] *= 0.65  # reduce sad bias slightly less aggressively
            pred_probs[2] *= 1.15  # boost neutral sensitivity
            pred_probs[1] *= 1.25  # boost happy sensitivity
            pred_probs[0] *= 1.05  # boost angry sensitivity
            
            # Re-normalize
            sum_p = np.sum(pred_probs)
            if sum_p > 0:
                pred_probs /= sum_p

            emotion_index = np.argmax(pred_probs)
            confidence = float(pred_probs[emotion_index]) * 100

            emotion = self.emotion_labels[
                emotion_index
            ]

            self.buffer.append(
                emotion
            )

            counts = {
                e: self.buffer.count(e)
                for e in set(self.buffer)
            }

            detected_emotion = max(
                counts,
                key=counts.get
            )

            cv2.putText(
                frame,
                f"{detected_emotion} ({confidence:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        return (
            frame,
            detected_emotion,
            confidence
        )