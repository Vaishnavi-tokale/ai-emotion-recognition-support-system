# remove_duplicates.py
from PIL import Image
import imagehash
from pathlib import Path
import shutil

images_dir = Path("E:/AI_Therapist_Project/Dataset/images")
dups_dir = Path("E:/AI_Therapist_Project/Dataset/duplicates")

dups_dir.mkdir(exist_ok=True)

hashes = {}
duplicates = []
print("🔍 Checking for duplicate images...")

for p in images_dir.rglob("*.*"):
    try:
        h = imagehash.average_hash(Image.open(p))
        if h in hashes:
            duplicates.append(p.name)
            shutil.move(str(p), str(dups_dir / p.name))
        else:
            hashes[h] = p.name
    except Exception:
        continue

print(f"\n✅ Duplicate detection complete. Duplicates moved: {len(duplicates)}")
if duplicates:
    print("Examples:", duplicates[:5])

