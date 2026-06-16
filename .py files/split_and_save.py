# split_and_save.py (low-memory version)
import numpy as np
from sklearn.model_selection import train_test_split

print("📂 Loading preprocessed data...")
X = np.load("Dataset/X.npy", mmap_mode='r')  # load without putting full array into RAM
y = np.load("Dataset/y.npy")

print("🔹 Splitting indices into train/test...")
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(np.zeros(len(y)), y):
    np.save("Dataset/train_idx.npy", train_idx)
    np.save("Dataset/test_idx.npy", test_idx)
print("✅ Indices split complete!")

# Now save smaller split arrays
print("💾 Saving splits to disk (chunked)...")
np.save("Dataset/y_train.npy", y[train_idx])
np.save("Dataset/y_test.npy", y[test_idx])
print("✅ Label splits saved successfully!")

