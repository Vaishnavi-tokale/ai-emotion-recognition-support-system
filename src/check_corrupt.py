# check_corrupt.py
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import shutil

# Define directories
images_dir = Path("E:/AI_Therapist_Project/Dataset/images")
corrupt_dir = Path("E:/AI_Therapist_Project/Dataset/corrupt_images")

corrupt_dir.mkdir(exist_ok=True)

corrupt = []
print("🔍 Checking for corrupt images...")

for p in images_dir.rglob("*.*"):
    try:
        with Image.open(p) as img:
            img.verify()  # quick check for corruption
    except (UnidentifiedImageError, OSError):
        corrupt.append(p.name)
        shutil.move(str(p), str(corrupt_dir / p.name))

print(f"\n✅ Completed! Corrupt images moved: {len(corrupt)}")
if corrupt:
    print("Examples:", corrupt[:5])
