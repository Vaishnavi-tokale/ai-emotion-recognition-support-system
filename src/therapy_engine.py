# ==========================================
# therapy_engine.py
# AI Therapist Recommendation Engine
# ==========================================

from therapy_database import therapy_recommendations, therapy_sessions


class TherapyEngine:

    def __init__(self):
        self.recommendations = therapy_recommendations
        self.sessions = therapy_sessions

    # --------------------------------------
    # Get Recommended Therapies
    # --------------------------------------
    def get_recommendations(self, emotion, gesture):

        key = (emotion.lower(), gesture.lower())

        if key in self.recommendations:
            return self.recommendations[key]

        # Default recommendations
        return [
            "Guided Breathing",
            "Relaxation Exercise",
            "Positive Encouragement"
        ]

    # --------------------------------------
    # Get Complete Therapy Session
    # --------------------------------------
    def get_session(self, therapy_name):

        if therapy_name in self.sessions:
            return self.sessions[therapy_name]

        return {
            "description": "Therapy not available.",
            "steps": ["Please try another activity."],
            "voice": "Sorry, I could not find this therapy."
        }

    # --------------------------------------
    # Generate Therapist Intro
    # --------------------------------------
    def generate_intro(self, emotion, gesture):

        emotion = emotion.capitalize()
        gesture = gesture.capitalize()

        return (
            f"I noticed that you may be feeling {emotion} "
            f"and your hand gesture indicates '{gesture}'. "
            f"I have prepared a few supportive activities. "
            f"Please choose the one that feels most comfortable."
        )

    # --------------------------------------
    # Print Recommendation Menu
    # --------------------------------------
    def display_options(self, emotion, gesture):

        options = self.get_recommendations(emotion, gesture)

        print("\n==============================")
        print("      AI THERAPIST")
        print("==============================\n")

        print(self.generate_intro(emotion, gesture))
        print("\nRecommended Support Activities:\n")

        for i, option in enumerate(options, start=1):
            print(f"{i}. {option}")

        print()

        return options

    # --------------------------------------
    # Execute Therapy
    # --------------------------------------
    def start_therapy(self, therapy_name):

        session = self.get_session(therapy_name)

        print("\n===================================")
        print(session["description"])
        print("===================================\n")

        for step_no, step in enumerate(session["steps"], start=1):
            print(f"Step {step_no}: {step}")

        return session["voice"]


# ==========================================
# Test Therapy Engine
# ==========================================

if __name__ == "__main__":

    engine = TherapyEngine()

    emotion = "sad"
    gesture = "help"

    options = engine.display_options(emotion, gesture)

    choice = int(input("Select Therapy (1-5): "))

    therapy = options[choice - 1]

    voice_text = engine.start_therapy(therapy)

    print("\nVoice Output:")
    print(voice_text)