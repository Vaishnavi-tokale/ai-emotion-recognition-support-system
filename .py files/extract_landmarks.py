import os
import cv2
import numpy as np
import mediapipe as mp

DATA_PATH = "filtered_videos"
SAVE_PATH = "landmark_sequences"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs(SAVE_PATH, exist_ok=True)

SEQUENCE_LENGTH = 40   # fixed sequence length

def extract_keypoints(results):
    keypoints = np.zeros(126)  # 63 per hand × 2 hands

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            for j, lm in enumerate(hand_landmarks.landmark):
                keypoints[i*63 + j*3 : i*63 + j*3 + 3] = [lm.x, lm.y, lm.z]

    return keypoints

for label in os.listdir(DATA_PATH):
    label_path = os.path.join(DATA_PATH, label)
    save_label_path = os.path.join(SAVE_PATH, label)
    os.makedirs(save_label_path, exist_ok=True)

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

        if len(sequence) > 0:
            sequence = np.array(sequence)

            # pad or trim to fixed length
            if len(sequence) < SEQUENCE_LENGTH:
                padding = np.zeros((SEQUENCE_LENGTH - len(sequence), 126))
                sequence = np.vstack((sequence, padding))
            else:
                sequence = sequence[:SEQUENCE_LENGTH]

            np.save(
                os.path.join(save_label_path, video_file.split(".")[0]),
                sequence
            )

print("Landmark extraction completed successfully!")