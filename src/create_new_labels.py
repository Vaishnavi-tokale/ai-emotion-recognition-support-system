import numpy as np
import pickle

labels = ['happy', 'sad', 'angry', 'help', 'yes', 'no']

# Save classes
np.save('label_classes.npy', labels)

# Create label map
label_map = {label: idx for idx, label in enumerate(labels)}

with open('label_map.pkl', 'wb') as f:
    pickle.dump(label_map, f)

print("New labels created:", label_map)