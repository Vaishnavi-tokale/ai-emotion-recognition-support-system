import json
from collections import defaultdict

JSON_PATH = "WLASL_v0.3.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

sign_count = defaultdict(int)

for item in data:
    sign_count[item["gloss"]] += len(item["instances"])

# Sort by frequency
sorted_signs = sorted(sign_count.items(), key=lambda x: x[1], reverse=True)

print("Top 20 most frequent signs:")
for sign, count in sorted_signs[:20]:
    print(f"{sign}: {count}")
