# ==========================================
# therapy_database.py
# AI Therapist - Therapy Database
# ==========================================

# -------------------------------------------------
# Therapy Recommendations
# (Emotion, Gesture) -> Recommended Activities
# -------------------------------------------------

therapy_recommendations = {

    ("sad", "help"): [
        "Guided Breathing",
        "Talk & Express",
        "Relaxation Exercise",
        "Positive Encouragement",
        "Mindfulness Session"
    ],

    ("sad", "yes"): [
        "Guided Breathing",
        "Positive Encouragement",
        "Gratitude Exercise"
    ],

    ("sad", "no"): [
        "Relaxation Exercise",
        "Mindfulness Session",
        "Talk & Express"
    ],

    ("angry", "help"): [
        "Anger Relaxation",
        "Guided Breathing",
        "Positive Thinking",
        "Mindfulness Session",
        "Talk & Express"
    ],

    ("angry", "yes"): [
        "Guided Breathing",
        "Relaxation Exercise",
        "Positive Thinking"
    ],

    ("angry", "no"): [
        "Mindfulness Session",
        "Calming Exercise",
        "Positive Encouragement"
    ],

    ("happy", "yes"): [
        "Gratitude Exercise",
        "Goal Planning",
        "Positive Conversation",
        "Inspirational Story"
    ],

    ("happy", "help"): [
        "Positive Conversation",
        "Goal Planning",
        "Gratitude Exercise"
    ],

    ("happy", "no"): [
        "Relaxation Exercise",
        "Positive Conversation"
    ],

    ("neutral", "help"): [
        "Wellness Check",
        "Talk & Express",
        "Guided Breathing"
    ],

    ("neutral", "yes"): [
        "Daily Reflection",
        "Goal Planning",
        "Positive Conversation"
    ],

    ("neutral", "no"): [
        "Relaxation Exercise",
        "Mindfulness Session"
    ],

    ("fear", "help"): [
        "Grounding Exercise",
        "Guided Breathing",
        "Positive Encouragement",
        "Mindfulness Session"
    ]
}

# -------------------------------------------------
# Therapy Sessions
# -------------------------------------------------

therapy_sessions = {

    "Guided Breathing": {

        "description":
        "A simple breathing exercise to help you relax.",

        "steps": [

            "Sit comfortably.",

            "Close your eyes if you feel comfortable.",

            "Breathe in slowly through your nose for 4 seconds.",

            "Hold your breath for 4 seconds.",

            "Exhale gently through your mouth for 6 seconds.",

            "Repeat this cycle five times."

        ],

        "voice":
        "Let's begin a breathing exercise. Sit comfortably. Inhale for four seconds. Hold. Exhale slowly. Repeat five times."

    },

    "Talk & Express": {

        "description":
        "Expressing thoughts can reduce emotional stress.",

        "steps": [

            "Take a deep breath.",

            "Share what is on your mind.",

            "There is no right or wrong answer.",

            "Your feelings are important."

        ],

        "voice":
        "You can tell me whatever you are feeling. I am here to listen."

    },

    "Relaxation Exercise": {

        "description":
        "A short body relaxation exercise.",

        "steps": [

            "Relax your shoulders.",

            "Unclench your jaw.",

            "Take slow breaths.",

            "Stretch your neck gently.",

            "Relax your hands."

        ],

        "voice":
        "Relax your shoulders. Take a deep breath. Slowly release the tension from your body."

    },

    "Positive Encouragement": {

        "description":
        "Receive supportive motivational messages.",

        "steps": [

            "Every difficult moment passes.",

            "You are stronger than you think.",

            "Take one step at a time.",

            "Believe in yourself."

        ],

        "voice":
        "You are stronger than you think. Every challenge can be overcome one step at a time."

    },

    "Mindfulness Session": {

        "description":
        "Focus on the present moment.",

        "steps": [

            "Look around you.",

            "Notice five things you can see.",

            "Notice four things you can touch.",

            "Notice three sounds.",

            "Take a slow breath."

        ],

        "voice":
        "Focus on the present moment. Take a deep breath and notice your surroundings."

    },

    "Gratitude Exercise": {

        "description":
        "Practice gratitude to improve positive emotions.",

        "steps": [

            "Think of one person you appreciate.",

            "Think of one achievement.",

            "Think of one thing that made you smile today."

        ],

        "voice":
        "Let's practice gratitude. Think of something positive that happened today."

    },

    "Goal Planning": {

        "description":
        "Plan a small goal for today.",

        "steps": [

            "Choose one simple goal.",

            "Break it into small steps.",

            "Complete the first step today."

        ],

        "voice":
        "Small goals create big achievements. Let's plan one goal together."

    },

    "Positive Conversation": {

        "description":
        "A friendly supportive conversation.",

        "steps": [

            "Tell me how your day has been.",

            "What made you smile today?",

            "Is there anything exciting you want to share?"

        ],

        "voice":
        "I'd love to hear about your day."

    },

    "Inspirational Story": {

        "description":
        "Read a short motivational story.",

        "steps": [

            "Every successful person faced failures.",

            "Persistence leads to growth.",

            "Keep believing in yourself."

        ],

        "voice":
        "Every great achievement begins with a small step."

    },

    "Grounding Exercise": {

        "description":
        "Reduce anxiety using the 5-4-3-2-1 method.",

        "steps": [

            "Name five things you can see.",

            "Four things you can touch.",

            "Three things you can hear.",

            "Two things you can smell.",

            "One thing you can taste."

        ],

        "voice":
        "Let's practice grounding together using your senses."

    },

    "Anger Relaxation": {

        "description":
        "Reduce anger with calming techniques.",

        "steps": [

            "Pause for a moment.",

            "Take three deep breaths.",

            "Count slowly from one to ten.",

            "Think before reacting."

        ],

        "voice":
        "Take a deep breath. Count slowly from one to ten. You are in control."

    },

    "Positive Thinking": {

        "description":
        "Replace negative thoughts with positive ones.",

        "steps": [

            "Notice one negative thought.",

            "Replace it with a realistic positive thought.",

            "Repeat the positive thought."

        ],

        "voice":
        "Let's replace negative thoughts with positive and realistic ones."

    },

    "Calming Exercise": {

        "description":
        "A short calming routine.",

        "steps": [

            "Sit comfortably.",

            "Take slow breaths.",

            "Relax your muscles.",

            "Stay calm."

        ],

        "voice":
        "Let's calm your mind together."

    },

    "Wellness Check": {

        "description":
        "Reflect on your current wellbeing.",

        "steps": [

            "Take a deep breath.",

            "Notice how you feel today.",

            "Think of one positive action you can take."

        ],

        "voice":
        "Let's take a moment to check how you are feeling."

    },

    "Daily Reflection": {

        "description":
        "Reflect on your day.",

        "steps": [

            "What went well today?",

            "What challenged you?",

            "What are you proud of?"

        ],

        "voice":
        "Reflection helps us learn and grow."
    }
}