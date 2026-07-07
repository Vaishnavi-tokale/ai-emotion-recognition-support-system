import pyttsx3

engine = pyttsx3.init()
engine.say("This is a test. If you hear this, text to speech is working.")
engine.runAndWait()
