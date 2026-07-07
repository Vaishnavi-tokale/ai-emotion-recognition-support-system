import json
from collections import defaultdict

JSON_PATH = "WLASL_v0.3.json"

TARGET_SIGNS = ["happy", "sad", "help", "angry", "pain", "anxiety", "yes", "no"]

with open(JSON_PATH, "r") as f:
    data = json.load(f)

sign_count = defaultdict(int)

for item in data:
    gloss = item["gloss"]
    if gloss in TARGET_SIGNS:
        sign_count[gloss] += len(item["instances"])

print("Video count per selected sign:")
for sign, count in sign_count.items():
    print(f"{sign}: {count}")
