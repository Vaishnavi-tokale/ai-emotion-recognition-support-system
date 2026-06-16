# handle_missing.py
from pathlib import Path
import pandas as pd
import shutil

data_dir = Path("E:/AI_Therapist_Project/Dataset")
images_dir = data_dir / "images"
labels_file = data_dir / "data.xlsx.csv"
nolabel_dir = data_dir / "no_label"
missing_dir = data_dir / "missing_label_ref"

nolabel_dir.mkdir(exist_ok=True)
missing_dir.mkdir(exist_ok=True)

print("🔍 Checking missing labels and files...")

df = pd.read_csv(labels_file)
df['filename'] = df['path'].apply(lambda p: Path(p).name)

files_on_disk = {p.name for p in images_dir.rglob("*.*")}

# 1️⃣ Images present but not listed in label file
not_in_labels = [f for f in files_on_disk if f not in set(df['filename'])]
print("Images present but have no label:", len(not_in_labels))
if not_in_labels:
    print("Examples:", not_in_labels[:5])
# Optional: move for manual labeling
# for f in not_in_labels:
#     shutil.move(str(images_dir / f), str(nolabel_dir / f))

# 2️⃣ Labels referencing missing files
missing_refs = [f for f in df['filename'] if f not in files_on_disk]
print("Label references missing on disk:", len(missing_refs))
if missing_refs:
    pd.DataFrame({'missing_files': missing_refs}).to_csv(data_dir / "missing_files_report.csv", index=False)
    print("✅ Report saved to missing_files_report.csv")
