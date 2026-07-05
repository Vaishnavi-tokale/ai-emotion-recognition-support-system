# ==========================================
# voice_engine.py
# AI Therapist Voice Engine
# ==========================================

import pyttsx3
import time


class VoiceEngine:

    def __init__(self):

        self.engine = pyttsx3.init()

        self.engine.setProperty("rate", 160)

        self.engine.setProperty("volume", 1.0)

        self.last_message = ""

        self.last_time = 0

        self.cooldown = 3

        print("✅ Voice Engine Loaded")

    # -------------------------------------
    # Speak Text
    # -------------------------------------
    def speak(self, text):

        current_time = time.time()

        if (
            text == self.last_message
            and current_time - self.last_time < self.cooldown
        ):
            return

        self.last_message = text

        self.last_time = current_time

        self.engine.say(text)

        self.engine.runAndWait()

    # -------------------------------------
    # Stop Speaking
    # -------------------------------------
    def stop(self):

        self.engine.stop()

    # -------------------------------------
    # Speak Therapy Introduction
    # -------------------------------------
    def speak_intro(self, intro):

        self.speak(intro)

    # -------------------------------------
    # Speak Therapy Session
    # -------------------------------------
    def speak_therapy(self, session):

        self.speak(session["voice"])

    # -------------------------------------
    # Speak Emotion
    # -------------------------------------
    def speak_emotion(self, emotion):

        message = f"I detected your emotion as {emotion}."

        self.speak(message)

    # -------------------------------------
    # Speak Gesture
    # -------------------------------------
    def speak_gesture(self, gesture):

        message = f"I detected your hand gesture as {gesture}."

        self.speak(message)