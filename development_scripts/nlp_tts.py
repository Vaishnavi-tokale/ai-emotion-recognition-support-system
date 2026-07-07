import random
import pyttsx3

# Force Windows speech engine
engine = pyttsx3.init(driverName='sapi5')
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

responses = {
    "happy": [
        "You look happy today. Keep smiling.",
        "It is great to see you feeling positive."
    ],
    "sad": [
        "I sense that you are feeling sad. I am here with you.",
        "It is okay to feel low sometimes."
    ],
    "angry": [
        "You seem upset. Try to relax and breathe slowly.",
        "Let's calm down together."
    ],
    "fear": [
        "I sense anxiety. You are safe right now.",
        "Please stay calm. Everything will be okay."
    ],
    "neutral": [
        "You seem calm. Let me know if you need help."
    ]
}

def respond_with_speech(emotion):
    """
    ONE response:
    - same text is printed
    - same text is spoken
    """
    text = random.choice(responses.get(emotion, ["I am here to support you."]))

    # TEXT OUTPUT
    print("Therapist:", text)

    # VOICE OUTPUT (same text)
    engine.say(text)
    engine.runAndWait()

    return text   # useful later for UI / database
