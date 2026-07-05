# ============================================================
# AI THERAPIST - MAIN STREAMLIT APPLICATION
# ============================================================

import sys
import os
import time
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------
# Ensure the script directory and parent directory are in the system path
script_dir = os.path.abspath(os.path.dirname(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import local detectors and engine
from emotion_detector import EmotionDetector
from gesture_detector import GestureDetector
from therapy_engine import TherapyEngine

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="AI Therapist",
    page_icon="🤖",
    layout="wide"
)

# Premium balanced theme styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #EEF2F6;
    --bg-secondary: #E2E8F0;
    --card-bg: rgba(255, 255, 255, 0.95);
    --accent: #2563EB;
    --accent-hover: #1D4ED8;
    --text-primary: #1F2937;
    --text-secondary: #4B5563;
    --success: #10B981;
    --danger: #EF4444;
}

.stApp {
    background: linear-gradient(135deg, #F1F5F9 0%, #E2E8F0 100%) !important;
    font-family: 'Outfit', sans-serif;
    color: var(--text-primary) !important;
}

/* Force dark text for readability on light background */
h1, h2, h3, h4, h5, h6, p, span, label, div[data-testid="stMarkdownContainer"] p {
    color: var(--text-primary) !important;
}

.glass-card {
    background: var(--card-bg) !important;
    border: 1px solid rgba(226, 232, 240, 0.8) !important;
    border-radius: 20px;
    padding: 26px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.glass-card:hover {
    box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.08);
}

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(to right, #1E3A8A, #2563EB, #6366F1);
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
    font-size: 1.15rem;
    margin-bottom: 30px;
    font-weight: 500;
}

.step-card {
    background: #F8FAFC;
    border-left: 5px solid var(--success);
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 10px;
    font-size: 1.05rem;
    font-weight: 500;
}

.voice-card {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 12px;
    padding: 15px;
    margin-top: 15px;
    margin-bottom: 15px;
    font-style: italic;
    color: #1E40AF !important;
}

.feedback-container {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 12px;
    padding: 15px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# BACKEND MODULE INITIALIZATION (CACHED)
# ============================================================
from tensorflow.keras.models import load_model

@st.cache_resource
def get_cached_emotion_model():
    return load_model("trained_models/emotion_model_v1.h5")

@st.cache_resource
def get_cached_gesture_model():
    return load_model("sign_lstm_model.h5")

@st.cache_resource
def get_cached_therapy_engine():
    return TherapyEngine()

# Safe loading of models & detectors on rerun (lightweight)
emotion_detector = None
gesture_detector = None
therapy_engine = None

try:
    therapy_engine = get_cached_therapy_engine()
except Exception as e:
    st.error(f"⚠️ Failed to load Therapy Engine. Error: {e}")

try:
    emotion_model = get_cached_emotion_model()
    emotion_detector = EmotionDetector(model=emotion_model)
except Exception as e:
    st.error(f"⚠️ Failed to load Emotion Detector. Verify paths and weights. Error: {e}")

try:
    gesture_model = get_cached_gesture_model()
    gesture_detector = GestureDetector(model=gesture_model)
except Exception as e:
    st.error(f"⚠️ Failed to load Gesture Detector. Verify paths and weights. Error: {e}")
# ============================================================
# INITIALIZE SESSION STATE
# ============================================================
if "camera_started" not in st.session_state:
    st.session_state.camera_started = False

if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

if "selected_emotion" not in st.session_state:
    st.session_state.selected_emotion = "Sad"

if "selected_gesture" not in st.session_state:
    st.session_state.selected_gesture = "Help"

if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = "Neutral"

if "detected_gesture" not in st.session_state:
    st.session_state.detected_gesture = "Unknown"

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "recommendations_triggered" not in st.session_state:
    st.session_state.recommendations_triggered = False

if "selected_therapy" not in st.session_state:
    st.session_state.selected_therapy = None

if "speak_text" not in st.session_state:
    st.session_state.speak_text = None

if "spoken_text" not in st.session_state:
    st.session_state.spoken_text = ""

if "feedback" not in st.session_state:
    st.session_state.feedback = None

if "detection_lock_time" not in st.session_state:
    st.session_state.detection_lock_time = 0.0

if "locked_gesture" not in st.session_state:
    st.session_state.locked_gesture = "Unknown"

# Stable detection tracking and persistent camera states
if "previous_emotion" not in st.session_state:
    st.session_state.previous_emotion = None

if "previous_gesture" not in st.session_state:
    st.session_state.previous_gesture = None

if "candidate_stable_emotion" not in st.session_state:
    st.session_state.candidate_stable_emotion = None

if "candidate_stable_gesture" not in st.session_state:
    st.session_state.candidate_stable_gesture = None

if "last_stable_change_time" not in st.session_state:
    st.session_state.last_stable_change_time = 0.0

if "initial_recommendation_done" not in st.session_state:
    st.session_state.initial_recommendation_done = False

if "cap" not in st.session_state:
    st.session_state.cap = None

st.markdown("<div class='main-title'>🤖 AI Therapist Support System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time Facial Emotion & Sign Gesture Recognition with Browser-Based Guided Therapy</div>", unsafe_allow_html=True)

# ============================================================
# VOICE SYNTHESIS HELPER (HTML5 SpeechSynthesis)
# ============================================================
speech_placeholder = st.empty()

def play_speech(text, placeholder=speech_placeholder):
    if text and text != st.session_state.spoken_text:
        js_code = f"""
        <script>
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            var utterance = new SpeechSynthesisUtterance({repr(text)});
            utterance.rate = 1.0;
            window.speechSynthesis.speak(utterance);
        }}
        </script>
        """
        with placeholder:
            components.html(js_code, height=0, width=0)
        st.session_state.spoken_text = text

# Play any pending speech at page load/rerun
if st.session_state.speak_text and st.session_state.speak_text != st.session_state.spoken_text:
    play_speech(st.session_state.speak_text, speech_placeholder)

def render_right_column(placeholder):
    with placeholder.container():
        # Display recommendations
        if st.session_state.recommendations_triggered:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("💡 Recommended Support Activities")
            
            emotion_val = st.session_state.selected_emotion if st.session_state.demo_mode else st.session_state.detected_emotion
            gesture_val = st.session_state.selected_gesture if st.session_state.demo_mode else st.session_state.detected_gesture
            
            intro_text = therapy_engine.generate_intro(emotion_val, gesture_val) if therapy_engine else ""
            st.info(intro_text)
            
            st.write("Click on any activity below to begin guided session:")
            
            for activity in st.session_state.recommendations:
                if st.button(f"▶ Start {activity}", key=f"start_btn_{activity}", use_container_width=True):
                    st.session_state.selected_therapy = activity
                    if therapy_engine:
                        session_data = therapy_engine.get_session(activity)
                        st.session_state.speak_text = session_data["voice"]
                    st.session_state.feedback = None
                    st.rerun()
                    
            st.markdown("</div>", unsafe_allow_html=True)

        # Display selected therapy session
        if st.session_state.selected_therapy:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader(f"🧘 Guided Session: {st.session_state.selected_therapy}")
            
            if therapy_engine:
                session_data = therapy_engine.get_session(st.session_state.selected_therapy)
                st.write(session_data["description"])
                
                st.markdown("#### ✅ Therapy steps:")
                for idx, step in enumerate(session_data["steps"], 1):
                    st.markdown(f"<div class='step-card'>{idx}. {step}</div>", unsafe_allow_html=True)
                
                st.markdown("#### ✅ Voice guidance text:")
                st.markdown(f"<div class='voice-card'>🔊 \"{session_data['voice']}\"</div>", unsafe_allow_html=True)
                
                # Feedback options
                st.markdown("#### ✅ Feedback options:")
                if st.session_state.feedback is None:
                    st.write("Was this activity helpful?")
                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1:
                        if st.button("😊 Highly Helpful", key="fb_highly_helpful", use_container_width=True):
                            st.session_state.feedback = "Highly Helpful"
                            st.rerun()
                    with col_f2:
                        if st.button("😐 Neutral", key="fb_neutral", use_container_width=True):
                            st.session_state.feedback = "Neutral"
                            st.rerun()
                    with col_f3:
                        if st.button("😢 Unhelpful", key="fb_unhelpful", use_container_width=True):
                            st.session_state.feedback = "Unhelpful"
                            st.rerun()
                else:
                    st.markdown(f"""
                    <div class='feedback-container'>
                        <strong>Feedback Recorded:</strong> {st.session_state.feedback}<br>
                        Thank you! Your feedback helps customize future sessions.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Therapy Engine data is unavailable.")
                
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# WORKFLOW LOGIC
# ============================================================

if not st.session_state.camera_started:
    # --------------------------------------------------------
    # 🏠 HOME SCREEN
    # --------------------------------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class='glass-card'>
            <h3 style='margin-top:0;'>🧘 Welcome to your AI Therapy Dashboard</h3>
            <p>This system uses computer vision and neural networks to help analyze your current emotional state and gestures to recommend therapeutic support.</p>
            <h4 style='margin-bottom:10px;'>💡 How it works:</h4>
            <ol>
                <li>Click <strong>Start Therapy</strong> below to initiate the session.</li>
                <li>The system will turn on your camera and identify facial emotions and hand signs.</li>
                <li>Click <strong>Recommend Therapy</strong> to analyze and select a guided breathing or relaxation exercise.</li>
                <li>Receive spoken guidance and step-by-step instructions.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("⚙️ Session Controls")
        
        # Demo Mode Selector
        st.session_state.demo_mode = st.checkbox(
            "🧪 Enable Demo Mode",
            value=st.session_state.demo_mode,
            help="Check this to test recommendation pathways manually without a webcam."
        )
        
        if st.session_state.demo_mode:
            st.info("Demo Mode Active: You can manually override emotion and gestures during the session.")
        else:
            st.success("Webcam Mode Active: Real-time neural networks will process your camera feed.")

        # Start button
        if st.button("▶ Start Therapy", use_container_width=True, type="primary"):
            st.session_state.camera_started = True
            st.session_state.recommendations = []
            st.session_state.recommendations_triggered = False
            st.session_state.selected_therapy = None
            st.session_state.speak_text = "Starting session. Please look at the camera." if not st.session_state.demo_mode else "Demo mode activated. Please select your parameters."
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # --------------------------------------------------------
    # 🧘 ACTIVE THERAPY SESSION
    # --------------------------------------------------------
    
    # Header bar with Stop Button
    col_hdr, col_stop = st.columns([4, 1])
    with col_hdr:
        st.markdown("### 🧘 Active Therapist Guidance Session")
    with col_stop:
        if st.button("⏹ Stop Therapy & Reset", use_container_width=True, type="secondary"):
            st.session_state.camera_started = False
            # Release persistent camera
            if st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                except BaseException:
                    pass
                st.session_state.cap = None
            st.session_state.recommendations = []
            st.session_state.recommendations_triggered = False
            st.session_state.selected_therapy = None
            st.session_state.speak_text = None
            st.session_state.spoken_text = ""
            st.session_state.feedback = None
            st.session_state.detected_emotion = "Neutral"
            st.session_state.detected_gesture = "Unknown"
            st.session_state.locked_emotion = "Neutral"
            st.session_state.locked_gesture = "Unknown"
            st.session_state.detection_lock_time = 0.0
            st.session_state.previous_emotion = None
            st.session_state.previous_gesture = None
            st.session_state.candidate_stable_emotion = None
            st.session_state.candidate_stable_gesture = None
            st.session_state.last_stable_change_time = 0.0
            st.session_state.initial_recommendation_done = False
            st.rerun()

    st.markdown("---")

    # Layout for active workspace
    col_input, col_output = st.columns([1, 1])

    with col_input:
        if st.session_state.demo_mode:
            # 🧪 Demo Input Controls
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🧪 Demo Mode Controls")
            st.write("Manually select inputs to test the Therapy Engine:")

            demo_emotion = st.selectbox(
                "Select Emotion:",
                options=["Sad", "Angry", "Happy", "Neutral", "Fear"],
                index=["Sad", "Angry", "Happy", "Neutral", "Fear"].index(st.session_state.selected_emotion)
            )
            st.session_state.selected_emotion = demo_emotion

            demo_gesture = st.selectbox(
                "Select Gesture:",
                options=["Help", "Yes", "No"],
                index=["Help", "Yes", "No"].index(st.session_state.selected_gesture)
            )
            st.session_state.selected_gesture = demo_gesture

            if therapy_engine:
                recs = therapy_engine.get_recommendations(
                    st.session_state.selected_emotion.lower(),
                    st.session_state.selected_gesture.lower()
                )
                st.session_state.recommendations = recs
                intro = therapy_engine.generate_intro(
                    st.session_state.selected_emotion,
                    st.session_state.selected_gesture
                )
                st.session_state.recommendations_triggered = True
                
                # Speak intro only if it's new
                if st.session_state.speak_text != intro and st.session_state.spoken_text != intro and st.session_state.selected_therapy is None:
                    st.session_state.speak_text = intro

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # 📷 Live Camera Feed Placeholder
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📷 Webcam Video Feed")
            frame_placeholder = st.empty()

            # Placeholders for live metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                emotion_metric_placeholder = st.empty()
            with col_m2:
                gesture_metric_placeholder = st.empty()

            st.markdown("---")
            if st.button("🔄 Scan Again / Re-Detect", use_container_width=True, type="primary"):
                st.session_state.detection_lock_time = 0.0
                st.session_state.recommendations_triggered = False
                st.session_state.selected_therapy = None
                st.session_state.feedback = None
                st.session_state.detected_emotion = "Neutral"
                st.session_state.detected_gesture = "Unknown"
                st.session_state.locked_emotion = "Neutral"
                st.session_state.locked_gesture = "Unknown"
                st.session_state.previous_emotion = None
                st.session_state.previous_gesture = None
                st.session_state.candidate_stable_emotion = None
                st.session_state.candidate_stable_gesture = None
                st.session_state.last_stable_change_time = 0.0
                st.session_state.initial_recommendation_done = False
                st.session_state.speak_text = None
                st.session_state.spoken_text = ""
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
            
    with col_output:
        output_placeholder = st.empty()
        render_right_column(output_placeholder)

    # --------------------------------------------------------
    # 📷 WEBCAM INFERENCE LOOP (Only runs when camera is ON & Demo Mode is OFF)
    # --------------------------------------------------------
    if not st.session_state.demo_mode:
        cap = st.session_state.cap
        if cap is None or not cap.isOpened():
            for attempt in range(5):
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    break
                time.sleep(0.3)
                
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                st.session_state.cap = cap
            else:
                frame_placeholder.error("Webcam is not accessible or locked by another application. Please verify your camera connection or enable Demo Mode.")
                st.session_state.camera_started = False
                st.session_state.cap = None
                st.stop()
                
        try:
            frame_count = 0
            while st.session_state.camera_started:
                ret, frame = cap.read()
                if not ret:
                    frame_placeholder.error("Webcam not accessible. Try checking if another application is using it or enable Demo Mode.")
                    break
                    
                frame = cv2.flip(frame, 1)
                current_time = time.time()
                frame_count += 1
                
                # Continuous detection: run model.predict() only once every 5 frames
                run_predict = (frame_count % 5 == 0)
                
                detected_emo = "Neutral"
                emotion_conf = 0.0
                if emotion_detector:
                    try:
                        frame, detected_emo, emotion_conf = emotion_detector.detect(frame, predict=run_predict)
                        if detected_emo != "Unknown":
                            st.session_state.detected_emotion = detected_emo
                    except Exception as ex:
                        print(f"Emotion detection error: {ex}")
                        
                detected_ges = "Unknown"
                gesture_conf = 0.0
                if gesture_detector:
                    try:
                        frame, detected_ges, gesture_conf = gesture_detector.detect(frame, predict=run_predict)
                        if detected_ges != "Unknown":
                            st.session_state.detected_gesture = detected_ges
                    except Exception as ex:
                        print(f"Gesture detection error: {ex}")
                        
                # Update UI metrics
                emotion_metric_placeholder.metric(
                    "Detected Emotion", 
                    st.session_state.detected_emotion.capitalize()
                )
                gesture_metric_placeholder.metric(
                    "Detected Gesture", 
                    st.session_state.detected_gesture.capitalize()
                )
                
                # Display processed frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")

                # Recommendation Update Logic:
                # 1. Pause automatic recommendation updates while a therapy session is active
                if st.session_state.selected_therapy is None:
                    stable_emotion = st.session_state.detected_emotion
                    stable_gesture = st.session_state.detected_gesture
                    
                    # 2. Check if we need to do the initial recommendation immediately
                    if not st.session_state.initial_recommendation_done:
                        st.session_state.previous_emotion = stable_emotion
                        st.session_state.previous_gesture = stable_gesture
                        st.session_state.candidate_stable_emotion = stable_emotion
                        st.session_state.candidate_stable_gesture = stable_gesture
                        st.session_state.last_stable_change_time = current_time
                        
                        if therapy_engine:
                            recs = therapy_engine.get_recommendations(
                                stable_emotion.lower(),
                                stable_gesture.lower()
                            )
                            st.session_state.recommendations = recs
                            st.session_state.recommendations_triggered = True
                            intro = therapy_engine.generate_intro(stable_emotion, stable_gesture)
                            st.session_state.speak_text = intro
                        st.session_state.initial_recommendation_done = True
                        st.rerun()
                        
                    # 3. Handle subsequent stable recommendation update tracking (2-3 seconds threshold)
                    elif (stable_emotion != st.session_state.previous_emotion or 
                          stable_gesture != st.session_state.previous_gesture):
                        
                        # If candidate changed, reset the stability timer
                        if (stable_emotion != st.session_state.candidate_stable_emotion or 
                            stable_gesture != st.session_state.candidate_stable_gesture):
                            st.session_state.candidate_stable_emotion = stable_emotion
                            st.session_state.candidate_stable_gesture = stable_gesture
                            st.session_state.last_stable_change_time = current_time
                        else:
                            # Candidate matches stable emotion/gesture. Check if it's been consistent for >= 2.5 seconds
                            if current_time - st.session_state.last_stable_change_time >= 2.5:
                                st.session_state.previous_emotion = stable_emotion
                                st.session_state.previous_gesture = stable_gesture
                                
                                if therapy_engine:
                                    recs = therapy_engine.get_recommendations(
                                        stable_emotion.lower(),
                                        stable_gesture.lower()
                                    )
                                    st.session_state.recommendations = recs
                                    st.session_state.recommendations_triggered = True
                                    intro = therapy_engine.generate_intro(stable_emotion, stable_gesture)
                                    st.session_state.speak_text = intro
                                    st.rerun()
                    else:
                        # Stable emotion and gesture returned to the committed states
                        st.session_state.candidate_stable_emotion = stable_emotion
                        st.session_state.candidate_stable_gesture = stable_gesture
                        st.session_state.last_stable_change_time = current_time

                # Wait for next frame (no explicit sleep to prevent frame queue buildup)
                pass
                
        finally:
            if not st.session_state.camera_started:
                if st.session_state.cap is not None:
                    try:
                        st.session_state.cap.release()
                    except BaseException:
                        pass
                    st.session_state.cap = None

