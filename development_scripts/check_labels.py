import json

TARGET_SIGNS = ["help", "sad", "angry", "happy", "yes", "no", "pain"]

with open("wlasl_processed/info.json", "r") as f:
    data = json.load(f)

available_signs = set(item["gloss"] for item in data)

print("\nChecking required signs:\n")


for sign in TARGET_SIGNS:
    if sign in available_signs:
        print(f"{sign} ✅ Found")
    else:
        print(f"{sign} ❌ NOT FOUND")