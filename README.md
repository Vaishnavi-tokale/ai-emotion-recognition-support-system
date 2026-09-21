# 🧠 AI-Based Emotion Recognition and Supportive Response System

### AI Therapist for Non-Verbal Communication

An AI-based multimodal communication system designed to support non-verbal users by combining **facial emotion recognition** and **hand gesture recognition** to understand non-verbal cues and generate supportive responses.

The system combines computer vision, deep learning, gesture recognition, and generative AI into an interactive real-time application.

---

## 🌟 Project Overview

Communication is not limited to spoken words. Facial expressions and hand gestures can communicate important emotional and conversational information, especially for people who have difficulty communicating verbally.

This project aims to provide a supportive AI interface that can:

- Detect emotions from facial expressions.
- Recognize predefined hand gestures/signs.
- Combine facial emotion and gesture information.
- Generate context-aware supportive responses.
- Provide optional voice feedback.
- Maintain a conversational session.
- Provide session insights and an exportable session summary.

The application is implemented as an interactive **Streamlit** application.

---

## 🎯 Objectives

The main objectives of this project are:

1. Develop a real-time facial emotion recognition system.
2. Recognize hand gestures using landmark-based sequence classification.
3. Combine multiple non-verbal communication signals.
4. Generate supportive responses based on the detected emotional state.
5. Provide voice feedback for generated responses.
6. Create an accessible interface for non-verbal communication support.
7. Maintain interaction history and provide session-level insights.

---

## ✨ Key Features

### 😊 Facial Emotion Recognition

The system analyzes facial expressions captured through the camera and predicts emotions such as:

- Happy
- Sad
- Angry
- Surprise
- Neutral

The facial emotion model is implemented using deep learning with TensorFlow/Keras.

---

### ✋ Hand Gesture Recognition

The application uses **MediaPipe Hands** to detect hand landmarks.

The extracted landmark sequences are passed to an LSTM-based gesture recognition model.

Supported gestures include examples such as:

- Yes
- No
- Help
- Happy
- Sad
- Angry
- Pain

---

### 🧠 Multimodal Emotion + Gesture Fusion

Instead of considering facial emotion and gesture independently, the system combines both signals.

For example:

```text
Facial Emotion: Sad
Gesture: Help

        ↓

Fused Emotional State

        ↓

Supportive AI Response
