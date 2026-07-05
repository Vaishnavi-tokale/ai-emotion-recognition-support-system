import cv2
import pickle
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


class GestureDetector:

    def __init__(self, model=None):

        # ---------------------------------
        # Load LSTM Model
        # ---------------------------------
        if model is not None:
            self.model = model
        else:
            self.model = load_model("sign_lstm_model.h5")

        # ---------------------------------
        # Load Label Map
        # ---------------------------------
        with open("label_map.pkl", "rb") as f:
            self.label_map = pickle.load(f)

        # ---------------------------------
        # MediaPipe Hands
        # ---------------------------------
        try:
            import mediapipe.python.solutions.hands as mp_hands
            import mediapipe.python.solutions.drawing_utils as mp_drawing
            self.mp_hands = mp_hands
            self.mp_drawing = mp_drawing
        except Exception:
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
            except Exception as e:
                raise AttributeError(f"MediaPipe solutions module could not be loaded: {e}")

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # ---------------------------------
        # Settings
        # ---------------------------------
        self.SEQUENCE_LENGTH = 30
        self.FEATURES = 63
        self.CONFIDENCE_THRESHOLD = 0.70

        self.sequence = []

        print("[OK] Gesture Detector Loaded")

    def detect(self, frame, predict=True):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Performance optimization: Resize frame for MediaPipe processing to reduce CPU/GPU usage and camera lag
        small_rgb = cv2.resize(rgb, (320, 240))

        results = self.hands.process(small_rgb)

        frame_landmarks = []

        gesture = "Unknown"
        confidence = 0

        # ---------------------------------
        # Hand Detection
        # ---------------------------------
        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                frame_landmarks.extend(
                    [lm.x, lm.y, lm.z]
                )

        # ---------------------------------
        # Force 63 Features
        # ---------------------------------
        if len(frame_landmarks) == 0:

            frame_landmarks = [0] * self.FEATURES

        elif len(frame_landmarks) > self.FEATURES:

            frame_landmarks = frame_landmarks[:self.FEATURES]

        elif len(frame_landmarks) < self.FEATURES:

            frame_landmarks.extend(
                [0] * (self.FEATURES - len(frame_landmarks))
            )

        # ---------------------------------
        # Build Sequence (Crucial to update on EVERY frame)
        # ---------------------------------
        self.sequence.append(frame_landmarks)

        if len(self.sequence) > self.SEQUENCE_LENGTH:
            self.sequence.pop(0)

        # If we do not need to run prediction, return early now that sequence is updated!
        if not predict:
            return frame, gesture, confidence

        # ---------------------------------
        # Predict Gesture
        # ---------------------------------
        if len(self.sequence) == self.SEQUENCE_LENGTH:

            input_data = np.expand_dims(
                np.array(self.sequence),
                axis=0
            )

            # Performance optimization: direct tensor execution bypasses high Keras overhead
            prediction = self.model(
                input_data,
                training=False
            ).numpy()[0]

            predicted_index = int(np.argmax(prediction))

            confidence = float(
                prediction[predicted_index]
            ) * 100

            if confidence >= self.CONFIDENCE_THRESHOLD * 100:

                gesture = self.label_map.get(
                    predicted_index,
                    "Unknown"
                )

                cv2.putText(
                    frame,
                    f"{gesture} ({confidence:.1f}%)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        return frame, gesture, confidence