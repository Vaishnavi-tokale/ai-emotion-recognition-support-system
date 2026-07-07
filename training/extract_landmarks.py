import os
import cv2
import numpy as np
import mediapipe as mp

# ==================================
# CONFIGURATION
# ==================================
DATA_PATH = "filtered_videos"
SAVE_PATH = "landmark_sequences"

SEQUENCE_LENGTH = 30
FEATURES = 63

# ==================================
# MEDIAPIPE HANDS
# ==================================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs(SAVE_PATH, exist_ok=True)

# ==================================
# EXTRACT SINGLE HAND LANDMARKS
# ==================================
def extract_keypoints(results):
    keypoints = np.zeros(FEATURES)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        for i, lm in enumerate(hand_landmarks.landmark):
            keypoints[i * 3:i * 3 + 3] = [lm.x, lm.y, lm.z]

    return keypoints

# ==================================
# PROCESS GESTURE CLASSES
# ==================================
VALID_LABELS = ["help", "no", "yes"]

for label in VALID_LABELS:

    label_path = os.path.join(DATA_PATH, label)

    if not os.path.exists(label_path):
        print(f"Folder not found: {label}")
        continue

    save_label_path = os.path.join(SAVE_PATH, label)
    os.makedirs(save_label_path, exist_ok=True)

    print(f"\nProcessing: {label}")

    for video_file in os.listdir(label_path):

        video_path = os.path.join(label_path, video_file)

        cap = cv2.VideoCapture(video_path)

        sequence = []

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(image)

            keypoints = extract_keypoints(results)

            sequence.append(keypoints)

        cap.release()

        if len(sequence) == 0:
            continue

        sequence = np.array(sequence)

        # Pad or trim sequence
        if len(sequence) < SEQUENCE_LENGTH:

            padding = np.zeros(
                (SEQUENCE_LENGTH - len(sequence), FEATURES)
            )

            sequence = np.vstack((sequence, padding))

        else:

            sequence = sequence[:SEQUENCE_LENGTH]

        save_file = os.path.join(
            save_label_path,
            os.path.splitext(video_file)[0]
        )

        np.save(save_file, sequence)

print("\nLandmark extraction completed successfully!")
print("Saved to landmark_sequences/")