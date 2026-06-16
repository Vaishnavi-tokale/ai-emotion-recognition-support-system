import pandas as pd
from pathlib import Path

# Path to your current filtered file
file_path = Path("E:/AI_Therapist_Project/Dataset/filtered_data.csv")

# Load and keep only required columns
df = pd.read_csv(file_path)
df = df[['path', 'label']]

# Remove duplicates and missing values if any
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Save the cleaned version
cleaned_path = file_path.parent / "filtered_data_clean.csv"
df.to_csv(cleaned_path, index=False)

print(f"✅ Clean CSV saved: {cleaned_path}")
print(f"Total valid entries: {len(df)}")
