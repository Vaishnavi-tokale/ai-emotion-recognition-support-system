# ==========================================
# session_manager.py
# ==========================================


class SessionManager:

    def __init__(self):

        self.reset()

    # --------------------------------------
    # Reset Session
    # --------------------------------------
    def reset(self):

        self.current_emotion = "Unknown"

        self.current_emotion_confidence = 0.0

        self.current_gesture = "Unknown"

        self.current_gesture_confidence = 0.0

        self.recommended_therapies = []

        self.selected_therapy = None

        self.therapy_started = False

        self.feedback = None

    # --------------------------------------
    # Emotion
    # --------------------------------------
    def update_emotion(self, emotion, confidence):

        self.current_emotion = emotion

        self.current_emotion_confidence = confidence

    # --------------------------------------
    # Gesture
    # --------------------------------------
    def update_gesture(self, gesture, confidence):

        self.current_gesture = gesture

        self.current_gesture_confidence = confidence

    # --------------------------------------
    # Therapy Recommendation
    # --------------------------------------
    def update_recommendations(self, therapies):

        self.recommended_therapies = therapies

    # --------------------------------------
    # Selected Therapy
    # --------------------------------------
    def select_therapy(self, therapy):

        self.selected_therapy = therapy

        self.therapy_started = True

    # --------------------------------------
    # Feedback
    # --------------------------------------
    def update_feedback(self, feedback):

        self.feedback = feedback

    # --------------------------------------
    # Get Current State
    # --------------------------------------
    def get_state(self):

        return {

            "emotion": self.current_emotion,

            "emotion_confidence": self.current_emotion_confidence,

            "gesture": self.current_gesture,

            "gesture_confidence": self.current_gesture_confidence,

            "recommended": self.recommended_therapies,

            "selected": self.selected_therapy,

            "started": self.therapy_started,

            "feedback": self.feedback
        }