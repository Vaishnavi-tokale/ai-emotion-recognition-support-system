import json

with open("E:/archive/nslt_100.json") as f:
    data = json.load(f)

labels = set()

for key in data:
    entry = data[key]
    label = entry.get('gloss') or entry.get('label') or entry.get('word')
    if label:
        labels.add(label)

print(sorted(labels))