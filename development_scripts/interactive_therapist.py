import speech_recognition as sr
import pyttsx3

# Initialize TTS
engine = pyttsx3.init(driverName='sapi5')
engine.setProperty('rate', 150)

def speak(text):
    print("Therapist:", text)
    engine.say(text)
    engine.runAndWait()

def generate_response(user_text):
    user_text = user_text.lower()

    if "sad" in user_text:
        return "I'm really sorry you're feeling sad. Do you want to share what happened?"
    elif "stress" in user_text or "stressed" in user_text:
        return "Stress can be overwhelming. Let's take a deep breath together."
    elif "happy" in user_text:
        return "That's wonderful to hear! What made you feel happy?"
    elif "angry" in user_text:
        return "It's okay to feel angry. Would you like to talk about what's bothering you?"
    else:
        return "I’m here to listen. Tell me more about how you're feeling."

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        user_text = recognizer.recognize_google(audio)
        print("You:", user_text)
        return user_text
    except:
        return None

# Conversation Loop
speak("Hello, I am here to talk with you. How are you feeling today?")

while True:
    user_input = listen()

    if user_input:
        if "exit" in user_input.lower():
            speak("Take care. I am always here for you.")
            break

        response = generate_response(user_input)
        speak(response)
