# face_check.py
import cv2
from pathlib import Path
import shutil

images_dir = Path("E:/AI_Therapist_Project/Dataset/images")
noface_dir = Path("E:/AI_Therapist_Project/Dataset/noface")

noface_dir.mkdir(exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

noface = []
print("🧠 Checking for images without faces...")

for p in images_dir.rglob("*.*"):
    img = cv2.imread(str(p))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        noface.append(p.name)
        shutil.move(str(p), str(noface_dir / p.name))

print(f"\n✅ Completed! Moved no-face images: {len(noface)}")
if noface:
    print("Examples:", noface[:5])
