import json
import os
import shutil

JSON_PATH = "WLASL_v0.3.json"
VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "sign_dataset"

TARGET_SIGNS = ["happy", "sad", "help", "angry", "pain", "yes", "no"]
MAX_VIDEOS_PER_SIGN = 10

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

for item in data:
    gloss = item["gloss"]

    if gloss in TARGET_SIGNS:
        dest_folder = os.path.join(OUTPUT_FOLDER, gloss)
        os.makedirs(dest_folder, exist_ok=True)

        instances = item["instances"][:MAX_VIDEOS_PER_SIGN]

        for instance in instances:
            video_id = instance["video_id"]
            video_name = f"{video_id}.mp4"

            source_path = os.path.join(VIDEO_FOLDER, video_name)

            if os.path.exists(source_path):
                shutil.copy(source_path, dest_folder)

print("Balanced dataset created successfully.")
