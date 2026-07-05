# Global configuration settings for the AI Therapist System

import os

# Model Paths (Using relative paths to E:\AI_Therapist_Project)
EMOTION_MODEL_PATH = os.path.join("Dataset", "emotion_model_vgg19.h5")
SIGN_MODEL_PATH = "sign_lstm_model.h5"
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
LABEL_MAP_PATH = "label_map.pkl"

# Model Preprocessing Dimensions
EMOTION_IMG_SIZE = (48, 48)  # Target shape for VGG19 input
EMOTION_CHANNELS = 3         # VGG19 expects (48, 48, 3) RGB

SIGN_SEQUENCE_LENGTH = 30    # sign_lstm_model expects (30, 63)
SIGN_NUM_FEATURES = 63

# Target Labels
EMOTION_LABELS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Rule-based fallback responses for the therapist
FALLBACK_RESPONSES = {
    "Happy": [
        "It's wonderful to see you smiling! What's making you feel good today?",
        "Your positive energy is great! Let's talk about what's bringing you joy.",
        "Seeing you happy is lovely. How can we build on this positive feeling?"
    ],
    "Sad": [
        "I'm here for you. It's completely okay to feel sad sometimes. What's on your mind?",
        "Please take a deep breath. You are not alone, and I am here to listen.",
        "I hear you, and it's okay to let these feelings out. Take all the time you need."
    ],
    "Angry": [
        "I hear that you're feeling angry or frustrated. Let's take a slow breath together.",
        "It's valid to feel angry. Let's try to gently unpack what's causing this tension.",
        "Take a moment to pause. I'm here to support you and work through this together."
    ],
    "Surprise": [
        "Oh! That seems to have caught you by surprise. What just happened?",
        "A moment of surprise! Let's take a breath and talk about it.",
        "You look surprised! How are you processing this unexpected moment?"
    ],
    "Neutral": [
        "I'm here with you. How has your day been going so far?",
        "Let check in with ourselves. How are you feeling overall right now?",
        "I'm ready whenever you are. What would you like to focus on today?"
    ]
}

DEFAULT_FALLBACK = "I am right here with you. Please feel free to share whatever you're feeling."

# ==========================================
# 🧘 INTERACTIVE THERAPEUTIC FLOWS DATA
# ==========================================

SUPPORT_MENUS = {
    "Sad_help": [
        "💬 Talk About Feelings",
        "🌱 Motivation Support",
        "✨ Positive Affirmation",
        "🧘 Stress Management"
    ],
    "Angry_help": [
        "🌬️ Breathing Exercise",
        "🧘 Calm Down Techniques",
        "🔥 Stress Relief Tips",
        "💬 Express Your Thoughts"
    ],
    "Happy_yes": [
        "🙏 Gratitude Reflection",
        "🎉 Celebrate Achievement",
        "🎯 Goal Setting",
        "✍️ Positive Journaling"
    ],
    "Neutral": [
        "📝 Daily Check-In",
        "💭 Self Reflection",
        "🔍 Mood Exploration",
        "🎯 Goal Planning"
    ]
}

QUICK_SUPPORT = [
    "💙 Calm Me Down",
    "🌬️ Quick Breathing",
    "✨ Positive Message",
    "🧘 Stress Relief"
]

LABEL_TO_FLOW = {
    "💬 Talk About Feelings": "talk_feelings",
    "💬 Express Your Thoughts": "talk_feelings",
    "🌱 Motivation Support": "motivation_support",
    "✨ Positive Affirmation": "positive_affirmation",
    "✨ Positive Message": "positive_affirmation",
    "🧘 Stress Management": "stress_management",
    "🧘 Stress Relief": "stress_management",
    "🔥 Stress Relief Tips": "stress_management",
    "🌬️ Breathing Exercise": "breathing_exercise",
    "🌬️ Quick Breathing": "breathing_exercise",
    "💙 Calm Me Down": "calm_down",
    "🧘 Calm Down Techniques": "calm_down",
    "🙏 Gratitude Reflection": "gratitude_reflection",
    "🎉 Celebrate Achievement": "celebrate_achievement",
    "🎯 Goal Setting": "goal_setting",
    "🎯 Goal Planning": "goal_setting",
    "✍️ Positive Journaling": "positive_journaling",
    "📝 Daily Check-In": "daily_check_in",
    "💭 Self Reflection": "self_reflection",
    "🔍 Mood Exploration": "mood_exploration"
}

THERAPEUTIC_FLOWS = {
    "positive_affirmation": {
        "title": "Positive Affirmation",
        "initial": "You are stronger than the challenges you are facing right now. Every difficult day is temporary and every small step forward matters.",
        "steps": [
            {
                "text": "Your feelings are valid, and it is okay to take things one day, or even one moment, at a time.",
                "options": ["✨ Another Affirmation", "🏠 Return to Main Menu"]
            },
            {
                "text": "You do not have to carry everything alone. You are worthy of peace, patience, and kindness from yourself.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    },
    "breathing_exercise": {
        "title": "Breathing Exercise",
        "initial": "Let's practice box breathing. Inhale deeply through your nose for 4 seconds... hold for 4 seconds... exhale slowly for 4 seconds... hold for 4 seconds. Let's do this together.",
        "steps": [
            {
                "text": "Take another slow breath in... feel your lungs expand... hold it... and release all the tension. You are doing great.",
                "options": ["🌬️ Next Cycle", "🍃 Deep Release", "🏠 Return to Main Menu"]
            },
            {
                "text": "Inhale a sense of calm... and as you exhale, imagine releasing all the weight you've been holding onto.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    },
    "stress_management": {
        "title": "Stress Management",
        "initial": "When stress rises, try tracing 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste. This is called the 5-4-3-2-1 grounding technique.",
        "steps": [
            {
                "text": "Focus on the physical space around you. Feel the chair supporting you, the floor beneath your feet. You are safe in this moment.",
                "options": ["🧘 Next Grounding Step", "🏠 Return to Main Menu"]
            },
            {
                "text": "Allow yourself to step away from your worries for just 5 minutes. Close your eyes and focus solely on the rhythm of your breathing.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    },
    "motivation_support": {
        "title": "Motivation Support",
        "initial": "Remember, progress is not linear. Even if you only take a tiny step today, like getting out of bed or taking a breath, that is a victory.",
        "steps": [
            {
                "text": "Think of one tiny task you can do next. It doesn't have to be perfect; it just needs to be done. I believe in you.",
                "options": ["🌱 Self-Compassion Reminder", "🏠 Return to Main Menu"]
            },
            {
                "text": "Be gentle with yourself. You are doing the best you can with the energy you have right now.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    },
    "gratitude_reflection": {
        "title": "Gratitude Reflection",
        "initial": "Let's shift our focus to something small but positive. Can you name one thing, no matter how small, that brought you a tiny bit of comfort today?",
        "steps": [
            {
                "text": "Think of a person, a pet, or a memory that makes you feel warm or safe. Let's hold onto that feeling for a moment.",
                "options": ["🍃 Sensory Gratitude", "🏠 Return to Main Menu"]
            },
            {
                "text": "Appreciate a physical comfort: a warm cup of tea, a soft blanket, or the quiet around you. These small things matter.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    },
    "goal_setting": {
        "title": "Goal Setting",
        "initial": "Let's set a micro-goal for today. A micro-goal is something small and achievable under 5 minutes, like drinking a glass of water or writing one sentence.",
        "steps": [
            {
                "text": "Break your micro-goal into its very first step. What is the absolute easiest action you can take to start?",
                "options": ["✅ Commitment Support", "🏠 Return to Main Menu"]
            },
            {
                "text": "Say it out loud or write it down. Committing to a small action helps build the momentum you need.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    },
    "daily_check_in": {
        "title": "Daily Check-In",
        "initial": "Daily check-ins help us align. Focus on your body and thoughts right now. What is one word that describes your state?",
        "steps": [
            {
                "text": "Whatever that word is, accept it without judgment. It is okay to feel exactly how you feel right now.",
                "options": ["💭 Self Reflection", "🏠 Return to Main Menu"]
            },
            {
                "text": "Take a moment to thank yourself for taking this time to pause. Self-awareness is the foundation of growth.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    },
    "mood_exploration": {
        "title": "Mood Exploration",
        "initial": "Mood exploration helps us see patterns. Let's look closer: are you feeling quiet and calm, or is there a subtle underlying worry?",
        "steps": [
            {
                "text": "Quiet and calm states are wonderful opportunities to recharge. Enjoy this peaceful baseline.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    },
    "calm_down": {
        "title": "Calm Down Flow",
        "initial": "It's okay to feel overwhelmed. Let's pause everything. Drop your shoulders, release your jaw, and take a long, slow exhale.",
        "steps": [
            {
                "text": "Let the tensed thoughts float by like clouds. You don't have to solve anything in this exact moment. Just exist.",
                "options": ["🌬️ Quick Breathing", "🏠 Return to Main Menu"]
            }
        ]
    },
    "talk_feelings": {
        "title": "Express Feelings",
        "initial": "Giving a name to our emotions reduces their power over us. If you could explain your feeling in a sentence, what would you say?",
        "steps": [
            {
                "text": "I hear you, and your perspective is completely valid. It takes strength to recognize and voice these thoughts.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    },
    "celebrate_achievement": {
        "title": "Celebrate Achievement",
        "initial": "Be proud of your accomplishments! Big or small, taking a moment to celebrate builds confidence and reinforces success.",
        "steps": [
            {
                "text": "Think about what actions helped you achieve this. Replicating those small habits is key to your continued progress.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    },
    "positive_journaling": {
        "title": "Positive Journaling",
        "initial": "Writing down positive details shifts our brain's filter. Write down three things that went well today, no matter how minor.",
        "steps": [
            {
                "text": "Reviewing these positive points when you feel down helps restore perspective. Keep practicing this simple habit.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    },
    "self_reflection": {
        "title": "Self Reflection",
        "initial": "Self-reflection connects us to our core values. Ask yourself: what is one choice you made recently that aligned with the person you want to be?",
        "steps": [
            {
                "text": "Acknowledge that alignment. Making value-based choices is how we build long-term contentment.",
                "options": ["🏠 Return to Main Menu"]
            }
        ]
    }
}

