# inspect_dataset.py
import pandas as pd
from pathlib import Path

# ✅ Update paths to match your folder
base_dir = Path(r"E:\AI_Therapist_Project\Dataset")
images_dir = base_dir / "dataset"      # Folder containing all images
labels_file = base_dir / "data.xlsx.csv"   # Excel file with emotion labels
print("Looking for:", labels_file)
print("Exists:", labels_file.exists())

# ✅ Read Excel file instead of CSV
df = pd.read_csv(labels_file)

print("✅ Dataset loaded successfully!\n")
print("Total label rows:", len(df))

print("\n🔹 Sample data:")
print(df.head())

# ✅ Check if 'emotion' column exists
if 'label' in df.columns:
    print("\nUnique emotions:", df['label'].unique())
    print("Total emotion classes:", df['label'].nunique())
else:
    print("\n⚠️ 'emotion' column not found. Columns available:")
    print(df.columns)
# Collect all image paths from folders
all_images = list(images_dir.glob("*/*"))
print("\n✅ Total images physically found:", len(all_images))

# Convert Path objects to relative string (for comparison)
image_names = {str(p.relative_to(images_dir)).replace("\\", "/") for p in all_images}

# Filter dataframe for existing images
df['exists'] = df['path'].isin(image_names)
existing_df = df[df['exists'] == True]

print("\n✅ Images available in both CSV and folder:", len(existing_df))
print(existing_df.head())

# Optional: save this filtered list to a new CSV
existing_df.to_csv(base_dir / "filtered_data.csv", index=False)
print("\n✅ Saved matched image list to 'filtered_data.csv'")


