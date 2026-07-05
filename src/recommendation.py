# ==========================================
# recommendation.py
# Recommendation Engine
# ==========================================

from therapy_engine import TherapyEngine


class RecommendationEngine:

    def __init__(self):

        self.engine = TherapyEngine()

    # ----------------------------------------
    # Generate Recommendations
    # ----------------------------------------
    def recommend(self, emotion, gesture):

        # Handle empty values
        if emotion is None:
            emotion = "neutral"

        if gesture is None:
            gesture = "help"

        emotion = emotion.lower()
        gesture = gesture.lower()

        therapies = self.engine.get_recommendations(
            emotion,
            gesture
        )

        return therapies

    # ----------------------------------------
    # Therapy Session
    # ----------------------------------------
    def get_session(self, therapy_name):

        return self.engine.get_session(
            therapy_name
        )

    # ----------------------------------------
    # Intro Message
    # ----------------------------------------
    def get_intro(self, emotion, gesture):

        return self.engine.generate_intro(
            emotion,
            gesture
        )