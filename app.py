import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from collections import deque
import time
import base64
import os
import threading
import random
import queue
import google.generativeai as genai
import config
from datetime import datetime

def clean_html(html_str):
    """Strips leading/trailing whitespaces per line in a multi-line string to prevent markdown preformatted block parsing."""
    return "\n".join([line.strip() for line in html_str.split("\n")])

EMOTION_EMOJIS = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Surprise": "😲",
    "Neutral": "😌"
}

SIGN_EMOJIS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "help": "🤝",
    "yes": "👍",
    "no": "👎",
    "pain": "🤕"
}

# Thread-safe queue for async responses
RESPONSE_QUEUE = queue.Queue()

# Fused fallback responses for rule-based therapist dialogues
FUSED_FALLBACK_RESPONSES = {
    ("Sad", "help"): [
        "I see you're feeling down and asking for help. Please know that I am here to support you. What can I help you work through?",
        "Feeling sad and seeking support is a very brave first step. Let's take a deep breath together. I'm here to listen."
    ],
    ("Sad", "no"): [
        "It looks like you are feeling down and expressing 'no'. It's completely okay to not want to talk or engage right now. I'm here whenever you're ready.",
        "I notice you're feeling sad and gesturing 'no'. I respect your boundaries. We can just sit in quiet reflection for a moment if that helps."
    ],
    ("Angry", "no"): [
        "I notice your frustration and the gesture 'no'. It's completely valid to feel angry and reject what's happening. Let's pause and reset.",
        "When anger rises, saying 'no' is a natural response. Let's take a slow, calming breath together to release some of this tension."
    ],
    ("Angry", "help"): [
        "I hear your frustration, and I see you are asking for help. Let's gently unpack what is causing this agitation. I am here to guide you.",
        "It sounds like you are feeling angry and need some support. Let's take a moment to step back and figure this out together."
    ],
    ("Happy", "yes"): [
        "It is wonderful to see you happy and receptive! What positive developments or thoughts would you like to share today?",
        "Your happy energy and positive gesture are wonderful to see. Let's build on this positive momentum!"
    ],
    ("Happy", "happy"): [
        "It's great to see your happiness shine through both your expression and sign! Tell me more about what's bringing you joy.",
        "Such positive energy! I'm delighted to share this happy moment with you. What's on your mind?"
    ],
    ("Neutral", "yes"): [
        "I see you're feeling calm and showing agreement. How can we make the most of this balanced state today?",
        "You look centered and ready to cooperate. What would you like to focus our discussion on?"
    ],
    ("Neutral", "no"): [
        "I see you're in a neutral state but gesturing 'no'. Let me know if there's something you're uncomfortable discussing today.",
        "A neutral expression combined with 'no' suggests you might be hesitant. Let's take things at whatever pace feels comfortable for you."
    ],
    ("Neutral", "help"): [
        "I notice you are feeling neutral and asking for help. Let's explore what guidance or support you need today.",
        "You look calm, but you're gesturing for help. I'm ready to assist you. What area should we work on?"
    ]
}

# ==========================================
# LOGIN SYSTEM
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Set page config
st.set_page_config(page_title="AI Therapist Support System", layout="wide", page_icon="🧠")

# ==========================================
# PREMIUM DESIGN SYSTEM & CSS (BALANCED BLUE-GRAY THEME)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #E4E9F0;
    --bg-secondary: #CAD3DD;
    --card-bg: rgba(255, 255, 255, 0.85);
    --accent: #4F73B8;
    --accent-hover: #3C5F9E;
    --text-primary: #1E293B;
    --text-secondary: #475569;
    --success: #348846;
    --success-hover: #2B733B;
    --danger: #C84646;
    --danger-hover: #AB3A3A;
    --shadow-soft: 0 8px 30px rgba(71, 85, 105, 0.08);
    --border-light: 1px solid rgba(71, 85, 105, 0.15);
}

/* Base Styles */
.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, #D9E2EC 100%) !important;
    font-family: 'Outfit', sans-serif;
    color: var(--text-primary) !important;
}

/* Force dark text for readability on light background */
h1, h2, h3, h4, h5, h6, p, span, label, div[data-testid="stMarkdownContainer"] p {
    color: var(--text-primary) !important;
}

/* Glassmorphic Cards */
.glass-card {
    background: var(--card-bg) !important;
    border: var(--border-light) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-soft) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    color: var(--text-primary) !important;
}
.glass-card:hover {
    border-color: rgba(79, 115, 184, 0.3) !important;
    box-shadow: 0 12px 35px 0 rgba(79, 115, 184, 0.12) !important;
    transform: translateY(-2px);
}

/* Title & Headers */
.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(to right, #3C5F9E, var(--accent), #6366F1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 5px;
    letter-spacing: -0.02em;
}
.subtitle {
    text-align: center;
    color: var(--text-secondary) !important;
    font-size: 1.05rem;
    margin-bottom: 30px;
    font-weight: 500;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid rgba(71, 85, 105, 0.2) !important;
}
section[data-testid="stSidebar"] h2 {
    color: #3C5F9E !important;
}
section[data-testid="stSidebar"] label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* Custom styled inputs in Sidebar */
section[data-testid="stSidebar"] div[data-baseweb="input"] {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(71, 85, 105, 0.3) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79, 115, 184, 0.25) !important;
}

/* Sidebar button (Clear Chat History) */
section[data-testid="stSidebar"] div.stButton > button {
    background: white !important;
    color: #3C5F9E !important;
    border: 1px solid rgba(60, 95, 158, 0.3) !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
    background: #F0F4FA !important;
    border-color: #3C5F9E !important;
    box-shadow: 0 4px 12px rgba(60, 95, 158, 0.08) !important;
}

/* Video image container - professional frame style */
div[data-testid="stImage"] img {
    border-radius: 20px !important;
    border: 3px solid rgba(79, 115, 184, 0.25) !important;
    box-shadow: 0 8px 30px rgba(71, 85, 105, 0.1) !important;
    transition: all 0.3s ease;
}
div[data-testid="stImage"] img:hover {
    border-color: rgba(79, 115, 184, 0.4) !important;
    box-shadow: 0 12px 35px rgba(71, 85, 105, 0.15) !important;
}

/* Button Customizations */
/* Start Camera Button (Column 1 inside Column 1) */
div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(1) div.stButton > button {
    background: linear-gradient(135deg, var(--success) 0%, #2B733B 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(52, 136, 70, 0.25) !important;
    transition: all 0.2s ease !important;
    min-height: 48px !important;
}
div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(1) div.stButton > button:hover {
    background: linear-gradient(135deg, #4CAF50 0%, var(--success-hover) 100%) !important;
    box-shadow: 0 6px 20px rgba(52, 136, 70, 0.35) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(1) div.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Stop Camera Button (Column 2 inside Column 1) */
div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(2) div.stButton > button {
    background: linear-gradient(135deg, var(--danger) 0%, #AB3A3A 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(200, 70, 70, 0.25) !important;
    transition: all 0.2s ease !important;
    min-height: 48px !important;
}
div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(2) div.stButton > button:hover {
    background: linear-gradient(135deg, #EF5350 0%, var(--danger-hover) 100%) !important;
    box-shadow: 0 6px 20px rgba(200, 70, 70, 0.35) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(2) div.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Login Card Button Override */
.glass-card div.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #3C5F9E 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(79, 115, 184, 0.2) !important;
    transition: all 0.3s ease !important;
    min-height: 44px !important;
}
.glass-card div.stButton > button:hover {
    background: linear-gradient(135deg, #6C8CFF 0%, #2B4E8C 100%) !important;
    box-shadow: 0 6px 20px rgba(79, 115, 184, 0.3) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

def login():
    st.markdown("<div class='glass-card' style='max-width: 450px; margin: 80px auto;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #3C5F9E; font-weight: 600; margin-bottom: 24px;'>🔐 AI Therapist Login</h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    login()
    st.stop()

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
if "run" not in st.session_state:
    st.session_state.run = False

if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True

if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False

if "raw_emotion_probs" not in st.session_state:
    st.session_state.raw_emotion_probs = {"Angry": 0.0, "Happy": 0.0, "Neutral": 0.0, "Sad": 0.0, "Surprise": 0.0}

if "current_response" not in st.session_state:
    st.session_state.current_response = "I am here with you. Please click 'Start' when you are ready."

if "last_spoken_emotion" not in st.session_state:
    st.session_state.last_spoken_emotion = ""

if "last_spoken_sign" not in st.session_state:
    st.session_state.last_spoken_sign = ""

if "last_spoken_fused" not in st.session_state:
    st.session_state.last_spoken_fused = ""

# Persistent stability state machine variables
if "active_emotion" not in st.session_state:
    st.session_state.active_emotion = "Neutral"
if "candidate_emotion" not in st.session_state:
    st.session_state.candidate_emotion = None
if "candidate_emotion_start_time" not in st.session_state:
    st.session_state.candidate_emotion_start_time = None

if "active_gesture" not in st.session_state:
    st.session_state.active_gesture = None
if "candidate_gesture" not in st.session_state:
    st.session_state.candidate_gesture = None
if "candidate_gesture_start_time" not in st.session_state:
    st.session_state.candidate_gesture_start_time = None

if "active_emotion_confidence" not in st.session_state:
    st.session_state.active_emotion_confidence = 1.0
if "active_gesture_confidence" not in st.session_state:
    st.session_state.active_gesture_confidence = 1.0

if "tts_queue" not in st.session_state:
    st.session_state.tts_queue = []

if "api_response_pending" not in st.session_state:
    st.session_state.api_response_pending = False

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = 0.0

if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = time.time()

if "therapy_flow_active" not in st.session_state:
    st.session_state.therapy_flow_active = False

if "therapy_flow_name" not in st.session_state:
    st.session_state.therapy_flow_name = ""

if "therapy_flow_step" not in st.session_state:
    st.session_state.therapy_flow_step = 0

if "therapy_fused_state" not in st.session_state:
    st.session_state.therapy_fused_state = ""

if "feedback_state" not in st.session_state:
    st.session_state.feedback_state = "none"

if "current_options" not in st.session_state:
    st.session_state.current_options = []

if "detected_emotions_log" not in st.session_state:
    st.session_state.detected_emotions_log = ["Neutral"]

if "detected_gestures_log" not in st.session_state:
    st.session_state.detected_gestures_log = []

if "used_paths_log" not in st.session_state:
    st.session_state.used_paths_log = []

if "exercises_completed" not in st.session_state:
    st.session_state.exercises_completed = 0

if "initial_detection_done" not in st.session_state:
    st.session_state.initial_detection_done = False

if "conversation_history" not in st.session_state:
    init_time = datetime.now().strftime("[%I:%M %p]")
    st.session_state.conversation_history = [
        {"role": "assistant", "content": "Hello! I am your AI therapist. How can I help you today?", "timestamp": init_time}
    ]

# ==========================================
# SIDEBAR / SETTINGS
# ==========================================
st.sidebar.markdown("<div style='text-align: center; padding: 10px;'><h2 style='color: #3C5F9E; margin: 0;'>⚙️ Control Panel</h2></div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
# API Key Input
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Gemini API key to enable intelligent AI response generation. If left blank, local rule-based responses will be used.", value=os.getenv("GEMINI_API_KEY", ""))

st.sidebar.markdown("### Settings")
st.session_state.voice_enabled = st.sidebar.toggle("🔊 Voice Feedback (TTS)", value=st.session_state.voice_enabled)

def generate_session_summary():
    start_dt = datetime.fromtimestamp(st.session_state.session_start_time)
    end_dt = datetime.now()
    duration_sec = int(time.time() - st.session_state.session_start_time)
    duration_str = f"{duration_sec // 60} minutes, {duration_sec % 60} seconds"
    
    # Chronological history lists
    emotions_list = st.session_state.detected_emotions_log
    gestures_list = st.session_state.detected_gestures_log
    paths_list = st.session_state.used_paths_log
    
    # Separate therapist messages and user choices from history
    therapist_msgs = []
    user_choices = []
    for msg in st.session_state.conversation_history:
        ts = msg.get("timestamp", "")
        content = msg["content"]
        if msg["role"] == "assistant":
            therapist_msgs.append(f"{ts} {content}")
        else:
            user_choices.append(f"{ts} {content}")
            
    # Insights calculations
    if emotions_list:
        most_emotion = max(set(emotions_list), key=emotions_list.count)
    else:
        most_emotion = "Neutral"
        
    if paths_list:
        most_support = max(set(paths_list), key=paths_list.count)
    else:
        most_support = "None"
        
    emotions_str = ", ".join(emotions_list) if emotions_list else "Neutral"
    gestures_str = ", ".join(gestures_list) if gestures_list else "None"
    paths_str = ", ".join(paths_list) if paths_list else "None"
    
    therapist_msg_str = "\n".join([f"- {m}" for m in therapist_msgs])
    user_choices_str = "\n".join([f"- {c}" for c in user_choices]) if user_choices else "- None"
    
    summary = f"""==================================================
🧠 AI THERAPIST SESSION SUMMARY
==================================================
[Session Date]
Date: {start_dt.strftime('%Y-%m-%d')}
Start Time: {start_dt.strftime('%I:%M:%S %p')}
End Time: {end_dt.strftime('%I:%M:%S %p')}

[Session Duration]
Duration: {duration_str}

[Emotion History]
Timeline: {emotions_str}

[Gesture History]
Timeline: {gestures_str}

[Selected Support Paths]
Paths: {paths_str}

[Therapist Messages]
{therapist_msg_str}

[User Choices]
{user_choices_str}

[Session Insights]
- Most Detected Emotion: {most_emotion}
- Most Used Support Path: {most_support}
- Total Interactions: {len(st.session_state.conversation_history) - 1}
- Exercises Completed: {st.session_state.exercises_completed}
==================================================
"""
    return summary

# Session Summary Export
summary_txt = generate_session_summary()
st.sidebar.download_button(
    label="📄 Export Session Summary",
    data=summary_txt,
    file_name=f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain",
    use_container_width=True
)

# Clear Conversation History
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    init_time = datetime.now().strftime("[%I:%M %p]")
    st.session_state.conversation_history = [
        {"role": "assistant", "content": "Hello! I am your AI therapist. How can I help you today?", "timestamp": init_time}
    ]
    st.session_state.current_response = "Hello! I am your AI therapist. How can I help you today?"
    st.session_state.last_spoken_emotion = ""
    st.session_state.last_spoken_sign = ""
    st.session_state.last_spoken_fused = ""
    st.session_state.last_response_time = 0.0
    st.session_state.api_response_pending = False
    st.session_state.active_emotion = "Neutral"
    st.session_state.candidate_emotion = None
    st.session_state.candidate_emotion_start_time = None
    st.session_state.active_gesture = None
    st.session_state.candidate_gesture = None
    st.session_state.candidate_gesture_start_time = None
    st.session_state.active_emotion_confidence = 1.0
    st.session_state.active_gesture_confidence = 1.0
    st.session_state.session_start_time = time.time()
    st.session_state.therapy_flow_active = False
    st.session_state.therapy_flow_name = ""
    st.session_state.therapy_flow_step = 0
    st.session_state.therapy_fused_state = ""
    st.session_state.feedback_state = "none"
    st.session_state.current_options = []
    st.session_state.detected_emotions_log = ["Neutral"]
    st.session_state.detected_gestures_log = []
    st.session_state.used_paths_log = []
    st.session_state.exercises_completed = 0
    st.session_state.initial_detection_done = False
    st.rerun()

# ==========================================
# HEADER
# ==========================================
st.markdown("<div class='main-title'>🧠 AI Therapist Support System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time Facial Emotion & Sign Gesture Recognition with Empathetic AI Feedback</div>", unsafe_allow_html=True)

# ==========================================
# LOAD MODELS (WITH CACHING & ERROR HANDLING)
# ==========================================
@st.cache_resource
def load_models():
    # Load VGG19 Emotion model
    emotion_model = load_model(config.EMOTION_MODEL_PATH, compile=False)
    # Load LSTM Sign model
    sign_model = load_model(config.SIGN_MODEL_PATH, compile=False)
    # Load Label map for sign model
    with open(config.LABEL_MAP_PATH, "rb") as f:
        label_map_raw = pickle.load(f)
    sign_labels_map = label_map_raw
    
    return emotion_model, sign_model, sign_labels_map

models_loaded = False
try:
    emotion_model, sign_model, label_map = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"❌ Failed to load models. Please make sure '{config.EMOTION_MODEL_PATH}' and '{config.SIGN_MODEL_PATH}' exist. Error: {e}")

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_PATH)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==========================================
# SYSTEM STATUS & INSIGHTS PANEL (SIDEBAR)
# ==========================================
st.sidebar.markdown("### 🔌 System Status")
camera_status = "🟢 Running" if st.session_state.run else "🔴 Stopped"
emotion_model_status = "🟢 VGG19 Loaded" if models_loaded else "🔴 Failed"
gesture_model_status = "🟢 LSTM Loaded" if models_loaded else "🔴 Failed"
gemini_status = "🟢 Gemini API" if api_key else "🟡 Fallback Rules"
voice_status = "🟢 Enabled 🔊" if st.session_state.voice_enabled else "🔴 Muted 🔇"

st.sidebar.markdown(clean_html(f"""
<div style='background: rgba(255, 255, 255, 0.65); padding: 12px; border-radius: 12px; border: 1px solid rgba(79, 115, 184, 0.2); font-size: 0.85rem; line-height: 1.6; margin-bottom: 15px;'>
    <strong>Camera:</strong> {camera_status}<br>
    <strong>Emotion Model:</strong> {emotion_model_status}<br>
    <strong>Gesture Model:</strong> {gesture_model_status}<br>
    <strong>Therapy Engine:</strong> {gemini_status}<br>
    <strong>Voice Feedback:</strong> {voice_status}
</div>
"""), unsafe_allow_html=True)

# Session Insights Card
duration_min = int((time.time() - st.session_state.session_start_time) / 60)
if st.session_state.detected_emotions_log:
    most_emotion = max(set(st.session_state.detected_emotions_log), key=st.session_state.detected_emotions_log.count)
else:
    most_emotion = "Neutral"
    
if st.session_state.used_paths_log:
    most_support = max(set(st.session_state.used_paths_log), key=st.session_state.used_paths_log.count)
else:
    most_support = "None"
    
st.sidebar.markdown("### 📊 Session Insights")
st.sidebar.markdown(clean_html(f"""
<div style='background: rgba(79, 115, 184, 0.08); padding: 12px; border-radius: 12px; border: 1px solid rgba(79, 115, 184, 0.25); font-size: 0.85rem; line-height: 1.6;'>
    <strong>Duration:</strong> {duration_min} min<br>
    <strong>Most Detected Emotion:</strong> {most_emotion}<br>
    <strong>Most Used Support:</strong> {most_support}<br>
    <strong>Total Interactions:</strong> {len(st.session_state.conversation_history) - 1}<br>
    <strong>Exercises Completed:</strong> {st.session_state.exercises_completed}
</div>
"""), unsafe_allow_html=True)

st.sidebar.markdown("---")

emotion_buffer = deque(maxlen=6)
sequence = []

# ==========================================
# RESPONSE GENERATOR (NON-BLOCKING THREAD)
# ==========================================
def get_fusion_state(emotion, sign):
    if not sign:
        return f"Feeling {emotion}"
    
    combos = {
        ("Sad", "help"): "Distressed & seeking support",
        ("Sad", "no"): "Withdrawn & feeling down",
        ("Angry", "no"): "Defensive & agitated",
        ("Angry", "help"): "Frustrated & needing guidance",
        ("Happy", "yes"): "Joyful & receptive",
        ("Happy", "happy"): "Elated & positive",
        ("Neutral", "yes"): "Calm & cooperative",
        ("Neutral", "no"): "Quiet & hesitant",
        ("Neutral", "help"): "Curious & seeking clarity",
    }
    
    return combos.get((emotion, sign.lower()), f"Feeling {emotion} & gesturing '{sign}'")

def generate_therapist_response_async(emotion, sign, api_key):
    """Spawns a background worker thread to generate therapist responses and posts to RESPONSE_QUEUE."""
    st.session_state.api_response_pending = True
    
    # Take copies of variables to avoid cross-thread session_state accesses
    history_slice = list(st.session_state.conversation_history[-6:])
    voice_enabled = st.session_state.voice_enabled
    
    def worker():
        try:
            response_text = ""
            
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    
                    # Construct context
                    history_str = ""
                    for turn in history_slice:
                        role = "User" if turn["role"] == "user" else "Therapist"
                        history_str += f"{role}: {turn['content']}\n"
                    
                    fusion_state = get_fusion_state(emotion, sign)
                    prompt = f"""You are a compassionate, empathetic, and supportive AI therapist.
The user is interacting with you in real-time.
Their current state: {fusion_state}

Here is the recent conversation history:
{history_str}

Please generate a brief, comforting, and natural therapeutic response (1-2 sentences) directly addressing both their facial emotion ({emotion}) and their gesture ({sign or 'None'}) in an integrated, meaningful way. Do not use placeholders, labels, or templates. Speak directly and warmly as their therapist.
"""
                    response = model.generate_content(prompt)
                    response_text = response.text.strip()
                except Exception as api_err:
                    print(f"Gemini API Error: {api_err}")
            
            # Fallback if no API key or call failed
            if not response_text:
                # Check for fused multimodal fallback response first
                fused_key = (emotion, sign.lower() if sign else "")
                if fused_key in FUSED_FALLBACK_RESPONSES:
                    response_text = random.choice(FUSED_FALLBACK_RESPONSES[fused_key])
                else:
                    responses = config.FALLBACK_RESPONSES.get(emotion, [config.DEFAULT_FALLBACK])
                    if sign:
                        response_text = f"I notice you are gesturing '{sign}' and feeling {emotion.lower()}. " + random.choice(responses)
                    else:
                        response_text = random.choice(responses)
            
            # Enqueue the response object
            RESPONSE_QUEUE.put({
                "status": "success",
                "emotion": emotion,
                "sign": sign,
                "response_text": response_text,
                "voice_enabled": voice_enabled
            })
            
        except Exception as e:
            print(f"Error in response worker: {e}")
            try:
                RESPONSE_QUEUE.put({
                    "status": "error",
                    "error_msg": str(e)
                })
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True).start()

# ==========================================
# UI GRID COLUMNS
# ==========================================
col_feed, col_stats = st.columns([1.3, 1])

# Left Column - Video Feed & Buttons
with col_feed:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: var(--accent); margin-bottom: 8px;'>📹 Live Video Feed</div>", unsafe_allow_html=True)
    frame_placeholder = st.empty()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ Start Therapist Camera", use_container_width=True):
            st.session_state.run = True
            st.session_state.api_response_pending = False
            st.session_state.initial_detection_done = False
            st.session_state.last_response_time = 0.0
            st.session_state.last_spoken_emotion = ""
            st.session_state.last_spoken_sign = ""
            st.session_state.last_spoken_fused = ""
            st.session_state.active_emotion = "Neutral"
            st.session_state.candidate_emotion = None
            st.session_state.candidate_emotion_start_time = None
            st.session_state.active_gesture = None
            st.session_state.candidate_gesture = None
            st.session_state.candidate_gesture_start_time = None
            st.session_state.active_emotion_confidence = 1.0
            st.session_state.active_gesture_confidence = 1.0
            st.session_state.raw_emotion_probs = {"Angry": 0.0, "Happy": 0.0, "Neutral": 0.0, "Sad": 0.0, "Surprise": 0.0}
            st.rerun()
    with col_btn2:
        if st.button("⏹ Stop Therapist Camera", use_container_width=True):
            st.session_state.run = False
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Right Column - Metrics and Interactive Chat
with col_stats:
    # Live Metrics card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: var(--accent); margin-bottom: 12px;'>📊 Live Recognition Metrics</div>", unsafe_allow_html=True)
    
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        emotion_placeholder = st.empty()
    with col_metric2:
        sign_placeholder = st.empty()
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # helper functions for chat rendering and option clicking
    def render_chat_feed(placeholder):
        html_lines = []
        html_lines.append("""
        <div style='height: 380px; overflow-y: auto; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 12px; border: 1px solid rgba(79, 115, 184, 0.1); margin-bottom: 5px;'>
        """)
        
        for msg in st.session_state.conversation_history:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")
            
            if role == "assistant":
                html_lines.append(f"""
                <div style='display: flex; gap: 12px; align-items: flex-start; margin-bottom: 15px;'>
                    <div style='background: #4F73B8; color: white; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.15rem; box-shadow: 0 4px 8px rgba(79, 115, 184, 0.2); flex-shrink: 0;'>
                        🤖
                    </div>
                    <div style='flex-grow: 1;'>
                        <div style='font-size: 0.7rem; color: #475569; margin-bottom: 3px; font-weight: 600;'>{timestamp} AI Therapist</div>
                        <div style='background: linear-gradient(135deg, #F1F5F9 0%, #E2E8F0 100%); padding: 12px 16px; border-radius: 4px 16px 16px 16px; border: 1px solid rgba(79, 115, 184, 0.15); box-shadow: 0 4px 15px rgba(71, 85, 105, 0.02); font-size: 0.95rem; line-height: 1.5; color: #1E293B;'>
                            {content}
                        </div>
                    </div>
                </div>
                """)
            else:
                html_lines.append(f"""
                <div style='display: flex; gap: 12px; align-items: flex-start; justify-content: flex-end; margin-bottom: 15px;'>
                    <div style='text-align: right;'>
                        <div style='font-size: 0.7rem; color: #475569; margin-bottom: 3px; font-weight: 600;'>{timestamp} You</div>
                        <div style='background: #4F73B8; padding: 12px 16px; border-radius: 16px 4px 16px 16px; box-shadow: 0 4px 15px rgba(79, 115, 184, 0.15); font-size: 0.95rem; line-height: 1.5; color: white;'>
                            {content}
                        </div>
                    </div>
                    <div style='background: #1E293B; color: white; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.15rem; box-shadow: 0 4px 8px rgba(30, 41, 59, 0.2); flex-shrink: 0;'>
                        👤
                    </div>
                </div>
                """)
                
        html_lines.append("</div>")
        full_html = "".join(html_lines)
        placeholder.markdown(clean_html(full_html), unsafe_allow_html=True)

    def handle_option_click(label):
        t_now = datetime.now().strftime("[%I:%M %p]")
        st.session_state.conversation_history.append({"role": "user", "content": label, "timestamp": t_now})
        
        # Process choices
        if label == "🏠 Return to Main Menu":
            st.session_state.therapy_flow_active = False
            st.session_state.therapy_flow_name = ""
            st.session_state.therapy_flow_step = 0
            st.session_state.feedback_state = "none"
            st.session_state.current_options = config.SUPPORT_MENUS.get(st.session_state.therapy_fused_state, config.SUPPORT_MENUS["Neutral"])
            
            reply = "Returning to main menu. I am monitoring your emotion and gesture states in real-time."
            st.session_state.conversation_history.append({"role": "assistant", "content": reply, "timestamp": t_now})
            st.session_state.current_response = reply
            if st.session_state.voice_enabled:
                st.session_state.tts_queue.append(reply)
                
        elif label == "👍 Yes":
            st.session_state.exercises_completed += 1
            st.session_state.feedback_state = "helpful"
            reply = "I'm glad this helped. Remember that small positive steps can make a meaningful difference."
            st.session_state.conversation_history.append({"role": "assistant", "content": reply, "timestamp": t_now})
            st.session_state.current_response = reply
            st.session_state.current_options = ["🌱 Continue Support", "🏠 Return to Main Menu"]
            if st.session_state.voice_enabled:
                st.session_state.tts_queue.append(reply)
                
        elif label == "👎 No":
            st.session_state.feedback_state = "not_helpful"
            reply = "That's okay. Different approaches work for different people. Let's try another form of support."
            st.session_state.conversation_history.append({"role": "assistant", "content": reply, "timestamp": t_now})
            st.session_state.current_response = reply
            st.session_state.current_options = ["🔄 Alternative Support Options", "🏠 Return to Main Menu"]
            if st.session_state.voice_enabled:
                st.session_state.tts_queue.append(reply)
                
        elif label in ["🌱 Continue Support", "🔄 Alternative Support Options"]:
            st.session_state.therapy_flow_active = False
            st.session_state.therapy_flow_name = ""
            st.session_state.therapy_flow_step = 0
            st.session_state.feedback_state = "none"
            st.session_state.current_options = config.SUPPORT_MENUS.get(st.session_state.therapy_fused_state, config.SUPPORT_MENUS["Neutral"])
            
            reply = "Please select another support category."
            st.session_state.conversation_history.append({"role": "assistant", "content": reply, "timestamp": t_now})
            st.session_state.current_response = reply
            if st.session_state.voice_enabled:
                st.session_state.tts_queue.append(reply)
                
        else:
            flow_key = config.LABEL_TO_FLOW.get(label, None)
            if flow_key:
                st.session_state.therapy_flow_active = True
                st.session_state.therapy_flow_name = flow_key
                st.session_state.therapy_flow_step = 0
                st.session_state.feedback_state = "none"
                st.session_state.used_paths_log.append(config.THERAPEUTIC_FLOWS[flow_key]["title"])
                
                reply = config.THERAPEUTIC_FLOWS[flow_key]["initial"]
                st.session_state.conversation_history.append({"role": "assistant", "content": reply, "timestamp": t_now})
                st.session_state.current_response = reply
                
                flow_data = config.THERAPEUTIC_FLOWS[flow_key]
                if flow_data["steps"]:
                    st.session_state.current_options = flow_data["steps"][0]["options"]
                else:
                    st.session_state.current_options = ["🏠 Return to Main Menu"]
                
                if st.session_state.voice_enabled:
                    st.session_state.tts_queue.append(reply)
            else:
                active_flow = st.session_state.therapy_flow_name
                flow_data = config.THERAPEUTIC_FLOWS.get(active_flow, None)
                if flow_data:
                    step_idx = st.session_state.therapy_flow_step
                    next_step_idx = step_idx + 1
                    
                    if step_idx < len(flow_data["steps"]):
                        step_data = flow_data["steps"][step_idx]
                        reply = step_data["text"]
                        st.session_state.conversation_history.append({"role": "assistant", "content": reply, "timestamp": t_now})
                        st.session_state.current_response = reply
                        st.session_state.therapy_flow_step = next_step_idx
                        
                        if next_step_idx < len(flow_data["steps"]):
                            st.session_state.current_options = flow_data["steps"][next_step_idx]["options"]
                        else:
                            reply_fb = "Was this helpful?"
                            st.session_state.conversation_history.append({"role": "assistant", "content": reply_fb, "timestamp": t_now})
                            st.session_state.current_options = ["👍 Yes", "👎 No"]
                            
                        if st.session_state.voice_enabled:
                            st.session_state.tts_queue.append(reply)

    # Dialogue Response card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: var(--accent); margin-bottom: 12px;'>💬 Therapy Session Chat</div>", unsafe_allow_html=True)
    response_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive options panel card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: var(--accent); margin-bottom: 12px;'>🔘 Interactive Support Options</div>", unsafe_allow_html=True)
    options_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    # Always-Available Emergency Support panel card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: var(--danger); margin-bottom: 12px;'>🆘 Quick Support</div>", unsafe_allow_html=True)
    quick_support_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# Hidden TTS Audio Placeholder
tts_placeholder = st.empty()

# Play speech from queue when camera is not running
if not st.session_state.run and st.session_state.tts_queue:
    text_to_speak = st.session_state.tts_queue.pop(0)
    js_code = f"""
    <script>
    if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        var utterance = new SpeechSynthesisUtterance({repr(text_to_speak)});
        utterance.rate = 1.05;
        window.speechSynthesis.speak(utterance);
    }}
    </script>
    """
    with tts_placeholder:
        components.html(js_code, height=0, width=0)

# Draw initial UI states
emotion_placeholder.markdown(clean_html("""
<div style='background: rgba(79, 115, 184, 0.06); padding: 16px; border-radius: 16px; border: 1px solid rgba(79, 115, 184, 0.15); box-shadow: 0 4px 15px rgba(71, 85, 105, 0.02);'>
    <div style='font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; color: #4F73B8; margin-bottom: 6px; text-transform: uppercase;'>Current Emotion</div>
    <div style='font-size: 1.5rem; font-weight: 600; color: #1E293B; display: flex; align-items: center;'>
        <span style='margin-right: 8px;'>😌</span> Neutral
    </div>
</div>
"""), unsafe_allow_html=True)

sign_placeholder.markdown(clean_html("""
<div style='background: rgba(79, 115, 184, 0.06); padding: 16px; border-radius: 16px; border: 1px solid rgba(79, 115, 184, 0.15); box-shadow: 0 4px 15px rgba(71, 85, 105, 0.02);'>
    <div style='font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; color: #4F73B8; margin-bottom: 6px; text-transform: uppercase;'>Sign Gesture</div>
    <div style='font-size: 1.5rem; font-weight: 600; color: #1E293B; display: flex; align-items: center;'>
        <span style='margin-right: 8px;'>✋</span> None
    </div>
</div>
"""), unsafe_allow_html=True)

render_chat_feed(response_placeholder)

# Draw option button grids
with options_placeholder.container():
    if st.session_state.run and not st.session_state.initial_detection_done:
        st.markdown(clean_html("""
        <div style='background: rgba(79, 115, 184, 0.06); padding: 15px; border-radius: 12px; border: 1px dashed rgba(79, 115, 184, 0.3); text-align: center; color: var(--text-secondary); font-size: 0.9rem;'>
            🔍 <strong>Analyzing camera feed...</strong><br>
            Please look at the camera and make an emotion expression/gesture to begin your session.
        </div>
        """), unsafe_allow_html=True)
    else:
        opts = st.session_state.current_options if st.session_state.current_options else config.SUPPORT_MENUS["Neutral"]
        cols_opt = st.columns(2)
        for idx, opt in enumerate(opts):
            col_target = cols_opt[idx % 2]
            with col_target:
                if st.button(opt, key=f"opt_btn_{opt}_{idx}", use_container_width=True):
                    handle_option_click(opt)
                    st.rerun()

with quick_support_placeholder.container():
    if st.session_state.run and not st.session_state.initial_detection_done:
        st.markdown(clean_html("""
        <div style='background: rgba(200, 70, 70, 0.04); padding: 15px; border-radius: 12px; border: 1px dashed rgba(200, 70, 70, 0.2); text-align: center; color: var(--text-secondary); font-size: 0.9rem;'>
            ⏳ Waiting for initial analysis...
        </div>
        """), unsafe_allow_html=True)
    else:
        cols_quick = st.columns(2)
        for idx, qs in enumerate(config.QUICK_SUPPORT):
            col_target = cols_quick[idx % 2]
            with col_target:
                if st.button(qs, key=f"qs_btn_{qs}_{idx}", use_container_width=True):
                    handle_option_click(qs)
                    st.rerun()

# ==========================================
# CAMERA PIPELINE LOOP
# ==========================================
if st.session_state.run and models_loaded:
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    frame_count = 0
    detected_emotion = st.session_state.active_emotion
    predicted_sign = st.session_state.active_gesture
    consecutive_noface_frames = 0
    
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera. Please verify camera connection.")
            break
            
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Check if there is an asynchronous response ready from the background worker thread
        try:
            while not RESPONSE_QUEUE.empty():
                res = RESPONSE_QUEUE.get_nowait()
                st.session_state.api_response_pending = False
                if res.get("status") == "success":
                    st.session_state.initial_detection_done = True
                    st.session_state.current_response = res["response_text"]
                    t_now = datetime.now().strftime("[%I:%M %p]")
                    st.session_state.conversation_history.append({"role": "user", "content": f"[State: {get_fusion_state(res['emotion'], res['sign'])}]", "timestamp": t_now})
                    st.session_state.conversation_history.append({"role": "assistant", "content": res["response_text"], "timestamp": t_now})
                    
                    if not st.session_state.therapy_flow_active:
                        fused_key = f"{res['emotion']}_{res['sign'].lower()}" if res['sign'] else res['emotion']
                        if fused_key not in config.SUPPORT_MENUS:
                            fused_key = "Neutral"
                        st.session_state.current_options = config.SUPPORT_MENUS.get(fused_key, config.SUPPORT_MENUS["Neutral"])
                        st.session_state.therapy_fused_state = fused_key
                        
                    if res["voice_enabled"]:
                        st.session_state.tts_queue.append(res["response_text"])
                else:
                    print(f"Response worker error: {res.get('error_msg')}")
        except queue.Empty:
            pass
            
        # Play speech from queue if any
        if st.session_state.tts_queue:
            text_to_speak = st.session_state.tts_queue.pop(0)
            js_code = f"""
            <script>
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var utterance = new SpeechSynthesisUtterance({repr(text_to_speak)});
                utterance.rate = 1.05;
                window.speechSynthesis.speak(utterance);
            }}
            </script>
            """
            with tts_placeholder:
                components.html(js_code, height=0, width=0)
                
        # Performance optimization: skip heavy inference processing on odd frames
        if frame_count % 2 != 0:
            # Draw temporary status overlays on display frame
            cv2.putText(frame, f"Emotion: {detected_emotion}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (79, 115, 184), 2)
            if predicted_sign:
                cv2.putText(frame, f"Gesture: {predicted_sign}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 70, 70), 2)
            frame_placeholder.image(frame, channels="BGR")
            time.sleep(0.01)
            continue
            
        # ==========================================
        # FACE & EMOTION PIPELINE (VGG19)
        # ==========================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load frontal faces at full resolution to ensure maximum cascade sensitivity
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_frame_emotion = "Neutral"
        current_frame_emotion_confidence = 1.0
        
        if len(faces) == 0:
            # If no face is detected, increment counter. Revert to Neutral after 15 consecutive frames.
            consecutive_noface_frames += 1
            if consecutive_noface_frames > 15:
                emotion_buffer.append("Neutral")
            st.session_state.raw_emotion_probs = {"Angry": 0.0, "Happy": 0.0, "Neutral": 0.0, "Sad": 0.0, "Surprise": 0.0}
        else:
            consecutive_noface_frames = 0
            for x, y, w, h in faces:
                # Draw high-visibility face frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (79, 115, 184), 2)
                
                x_min, y_min = max(0, x), max(0, y)
                w_max = min(w, frame.shape[1] - x_min)
                h_max = min(h, frame.shape[0] - y_min)
                
                if w_max > 20 and h_max > 20:
                    face_crop = frame[y_min:y_min+h_max, x_min:x_min+w_max]
                    # Preprocess to match VGG19 color training format: RGB, 48x48, normalize 255.0
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, config.EMOTION_IMG_SIZE)
                    face_normalized = face_resized.astype('float32') / 255.0
                    face_input = np.expand_dims(face_normalized, axis=0)
                    
                    try:
                        # Direct tensor execution is much faster than model.predict
                        pred = emotion_model(face_input, training=False).numpy()[0]
                        
                        # Save raw emotion probabilities to session state
                        st.session_state.raw_emotion_probs = {
                            "Angry": float(pred[0]),
                            "Happy": float(pred[1]),
                            "Neutral": float(pred[2]),
                            "Sad": float(pred[3]),
                            "Surprise": float(pred[4])
                        }
                        
                        # Log raw predictions to terminal
                        print(f"Angry: {int(pred[0]*100)}%")
                        print(f"Happy: {int(pred[1]*100)}%")
                        print(f"Neutral: {int(pred[2]*100)}%")
                        print(f"Sad: {int(pred[3]*100)}%")
                        print(f"Surprise: {int(pred[4]*100)}%")
                        print(f"Predicted Emotion: {config.EMOTION_LABELS[np.argmax(pred)]}")
                        
                        confidence = np.max(pred)
                        if confidence >= 0.45:
                            current_frame_emotion = config.EMOTION_LABELS[np.argmax(pred)]
                            emotion_buffer.append(current_frame_emotion)
                            current_frame_emotion_confidence = confidence
                    except Exception as e:
                        pass
                    
                    # Draw label above box
                    cv2.putText(frame, current_frame_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (79, 115, 184), 2)
                    
        # Apply temporal smoothing via majority vote (at least 4 frames available)
        if len(emotion_buffer) >= 4:
            smoothed_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
        else:
            smoothed_emotion = current_frame_emotion
            
        # Update active emotion directly without dwell-time filters
        if smoothed_emotion != st.session_state.active_emotion:
            st.session_state.active_emotion = smoothed_emotion
            if not st.session_state.detected_emotions_log or st.session_state.detected_emotions_log[-1] != smoothed_emotion:
                st.session_state.detected_emotions_log.append(smoothed_emotion)
                
        if len(faces) > 0:
            st.session_state.active_emotion_confidence = current_frame_emotion_confidence
        else:
            st.session_state.active_emotion_confidence = 1.0
            
        detected_emotion = st.session_state.active_emotion
            
        # ==========================================
        # HANDS & SIGN PIPELINE (LSTM)
        # ==========================================
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        keypoints = []
        predicted_sign = None
        
        # Extract landmarks coordinates internally (WITHOUT drawing skeletons on frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
                    
        # Pad coordinates to match the 63 features (21 landmarks * 3 coords)
        keypoints = keypoints[:config.SIGN_NUM_FEATURES] + [0.0] * (config.SIGN_NUM_FEATURES - len(keypoints))
        sequence.append(keypoints)
        
        # Maintain sequence window of length 30
        if len(sequence) > config.SIGN_SEQUENCE_LENGTH:
            sequence.pop(0)
            
        raw_gesture = None
        current_frame_gesture_confidence = 1.0
        if len(sequence) == config.SIGN_SEQUENCE_LENGTH:
            # Only predict if hands are actively detected in this frame
            if result.multi_hand_landmarks:
                try:
                    pred_sign_seq = np.expand_dims(sequence, axis=0)
                    # Use direct tensor call for optimal camera frame rate
                    pred = sign_model(pred_sign_seq, training=False).numpy()[0]
                    predicted_class = np.argmax(pred)
                    confidence = pred[predicted_class]
                    if confidence >= 0.80:  # Gated at >= 80% confidence
                        raw_gesture = label_map.get(predicted_class, None)
                        current_frame_gesture_confidence = confidence
                except Exception as e:
                    pass
                    
        # Temporal consistency dwell-time filter (0.8 seconds check) using session state
        if raw_gesture != st.session_state.active_gesture:
            if raw_gesture == st.session_state.candidate_gesture:
                if st.session_state.candidate_gesture_start_time is not None:
                    elapsed = time.time() - st.session_state.candidate_gesture_start_time
                    if elapsed >= 0.8:  # Consistent for 0.8 seconds
                        st.session_state.active_gesture = st.session_state.candidate_gesture
                        st.session_state.candidate_gesture = None
                        st.session_state.candidate_gesture_start_time = None
                        st.session_state.active_gesture_confidence = current_frame_gesture_confidence
                        if st.session_state.active_gesture:
                            if not st.session_state.detected_gestures_log or st.session_state.detected_gestures_log[-1] != st.session_state.active_gesture:
                                st.session_state.detected_gestures_log.append(st.session_state.active_gesture)
            else:
                st.session_state.candidate_gesture = raw_gesture
                st.session_state.candidate_gesture_start_time = time.time()
        else:
            st.session_state.candidate_gesture = None
            st.session_state.candidate_gesture_start_time = None
            if raw_gesture is not None:
                st.session_state.active_gesture_confidence = current_frame_gesture_confidence
            else:
                st.session_state.active_gesture_confidence = 1.0
            
        predicted_sign = st.session_state.active_gesture
                
        # ==========================================
        # RESPONSE TRIGGER & MULTIMODAL FUSION
        # ==========================================
        current_fused = get_fusion_state(detected_emotion, predicted_sign)
        
        # Clear hand gesture lock when hand leaves the frame
        if not predicted_sign:
            st.session_state.last_spoken_sign = ""
            
        # Calculate response pacing cooldown
        current_time = time.time()
        time_since_last_response = current_time - st.session_state.last_response_time
        
        if not st.session_state.therapy_flow_active and not st.session_state.api_response_pending and time_since_last_response > 9.0:
            if current_fused != st.session_state.last_spoken_fused:
                generate_therapist_response_async(detected_emotion, predicted_sign, api_key)
                st.session_state.last_response_time = current_time
                st.session_state.last_spoken_fused = current_fused
                st.session_state.last_spoken_emotion = detected_emotion
                st.session_state.last_spoken_sign = predicted_sign if predicted_sign else ""
                    
        # ==========================================
        # RENDER FRAME & UPDATE METRICS PLACEHOLDERS
        # ==========================================
        frame_placeholder.image(frame, channels="BGR")
        
        # Update Recognition Metrics with beautiful blue-gray cards
        emotion_emoji = EMOTION_EMOJIS.get(detected_emotion, "😊")
        emotion_placeholder.markdown(clean_html(f"""
        <div style='background: rgba(79, 115, 184, 0.06); padding: 16px; border-radius: 16px; border: 1px solid rgba(79, 115, 184, 0.15); box-shadow: 0 4px 15px rgba(71, 85, 105, 0.02);'>
            <div style='font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; color: #4F73B8; margin-bottom: 6px; text-transform: uppercase;'>Current Emotion</div>
            <div style='font-size: 1.5rem; font-weight: 600; color: #1E293B; display: flex; align-items: center;'>
                <span style='margin-right: 8px;'>{emotion_emoji}</span> Emotion: {detected_emotion}
            </div>
        </div>
        """), unsafe_allow_html=True)
        
        sign_label_str = predicted_sign if predicted_sign else "None"
        sign_emoji = SIGN_EMOJIS.get(sign_label_str.lower(), "✋") if predicted_sign else "✋"
        
        sign_placeholder.markdown(clean_html(f"""
        <div style='background: rgba(79, 115, 184, 0.06); padding: 16px; border-radius: 16px; border: 1px solid rgba(79, 115, 184, 0.15); box-shadow: 0 4px 15px rgba(71, 85, 105, 0.02);'>
            <div style='font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; color: #4F73B8; margin-bottom: 6px; text-transform: uppercase;'>Sign Gesture</div>
            <div style='font-size: 1.5rem; font-weight: 600; color: #1E293B; display: flex; align-items: center;'>
                <span style='margin-right: 8px;'>{sign_emoji}</span> Gesture: {sign_label_str.capitalize()}
            </div>
        </div>
        """), unsafe_allow_html=True)
            
        render_chat_feed(response_placeholder)
        
        time.sleep(0.01)
        
    cap.release()