import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from collections import deque
import time
from gtts import gTTS
import base64
import os

# =========================
# LOGIN SYSTEM
# =========================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 AI Therapist Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(page_title="AI Therapist", layout="wide")

# =========================
# PREMIUM CSS
# =========================

st.markdown("""
<style>
body { background-color: #0f172a; }
h1 { text-align: center; color: #38bdf8; }
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}
.metric { font-size: 22px; color: #f1f5f9; }
.label { font-size: 14px; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown("<h1>🧠 AI Therapist System</h1>", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("""
- Emotion Detection
- Sign Recognition
- Voice Feedback
""")

# =========================
# SESSION STATE
# =========================

if "run" not in st.session_state:
    st.session_state.run = False

if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = ""

if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():
    emotion_model = load_model("model.hdf5", compile=False)
    sign_model = load_model("sign_lstm_model.h5")

    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)

    return emotion_model, sign_model, {v: k for k, v in label_map.items()}

emotion_model, sign_model, label_map = load_models()

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)

emotion_buffer = deque(maxlen=12)
sequence = []
SEQUENCE_LENGTH = 30

# =========================
# UI LAYOUT
# =========================

col1, col2 = st.columns([2,1])

frame_placeholder = col1.empty()
emotion_text = col2.empty()
sign_text = col2.empty()
response_text = col2.empty()

# =========================
# BUTTONS
# =========================

col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button("▶ Start"):
        st.session_state.run = True

with col_btn2:
    if st.button("⏹ Stop"):
        st.session_state.run = False

with col_btn3:
    st.session_state.voice_enabled = st.toggle("🔊 Voice", value=True)

# =========================
# VOICE FUNCTION
# =========================

def speak(text):
    try:
        file = "audio.mp3"
        gTTS(text=text).save(file)
        audio = open(file, "rb").read()
        st.markdown(
            f'<audio autoplay src="data:audio/mp3;base64,{base64.b64encode(audio).decode()}"></audio>',
            unsafe_allow_html=True
        )
        os.remove(file)
    except:
        pass

# =========================
# RESPONSE
# =========================

def therapy_response(e):
    return {
        "Happy":"You seem happy!",
        "Sad":"Take a deep breath.",
        "Angry":"Relax yourself.",
        "Fear":"You are safe.",
        "Distress":"I am here for you."
    }.get(e,"I'm here with you.")

# =========================
# CAMERA LOOP
# =========================

if st.session_state.run:

    cap = cv2.VideoCapture(0)
    cap.set(3,480)
    cap.set(4,360)

    frame_count = 0

    for _ in range(600):

        if not st.session_state.run:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame,1)
        frame_count += 1

        # Skip frames (performance boost)
        if frame_count % 2 != 0:
            frame_placeholder.image(frame, channels="BGR")
            continue

        # =========================
        # FACE EMOTION
        # =========================

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        detected_emotion = "Neutral"

        for x,y,w,h in faces:
            face = cv2.resize(gray[y:y+h,x:x+w],(64,64))/255
            face = face.reshape(1,64,64,1)
            pred = emotion_model.predict(face,verbose=0)
            emotion = emotion_labels[np.argmax(pred)]
            emotion_buffer.append(emotion)

        if len(emotion_buffer)>5:
            detected_emotion = max(set(emotion_buffer),key=emotion_buffer.count)

        # =========================
        # HAND + SIGN
        # =========================

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        keypoints = []
        predicted_sign = None

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                for lm in hand.landmark:
                    keypoints.extend([lm.x,lm.y,lm.z])

        keypoints = keypoints[:63] + [0]*(63-len(keypoints))
        sequence.append(keypoints)

        if len(sequence)>30:
            sequence.pop(0)

        if len(sequence)==30:
            try:
                pred = sign_model.predict(np.expand_dims(sequence,0),verbose=0)
                predicted_sign = label_map[np.argmax(pred)]
            except:
                pass

        # =========================
        # FINAL OUTPUT
        # =========================

        final_emotion = detected_emotion
        response = therapy_response(final_emotion)

        # Voice sync
        if st.session_state.voice_enabled:
            if final_emotion != st.session_state.last_spoken:
                speak(response)
                st.session_state.last_spoken = final_emotion

        # =========================
        # UI DISPLAY
        # =========================

        frame_placeholder.image(frame, channels="BGR")

        emotion_text.markdown(f"""
        <div class="card">
        <div class="label">Emotion</div>
        <div class="metric">😊 {final_emotion}</div>
        </div>
        """, unsafe_allow_html=True)

        sign_text.markdown(f"""
        <div class="card">
        <div class="label">Sign</div>
        <div class="metric">✋ {predicted_sign}</div>
        </div>
        """, unsafe_allow_html=True)

        response_text.markdown(f"""
        <div class="card">
        <div class="label">Response</div>
        <div class="metric">💬 {response}</div>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(0.03)

    cap.release()