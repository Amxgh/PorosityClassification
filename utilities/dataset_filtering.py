import csv
import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


directory = "./data/"

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

X_values = []
y_values = []
for (root, dirs, files) in os.walk(directory, topdown=True):
    for file in files:
        df = pd.read_csv(os.path.join(root, file), index_col=None, header=None)
        df = df.dropna(axis=1, how='any')
        numpy_df = df.to_numpy()

        max = np.max(numpy_df)

        threshold = max - 200
        if np.count_nonzero(numpy_df > threshold) > 4000:
            continue

        X_values.append(numpy_df)
        y_value = 0 if "unhealthy" in root else 1
        y_values.append(y_value)

        i = 0
        # print(type(X_values), len(X_values), X_values[0].shape if len(X_values) > 0 else "Empty")
        numpy_df = np.expand_dims(numpy_df, axis=-1)  # Adds channel dimension (480, 764, 1)
        numpy_df = np.expand_dims(numpy_df, axis=0)  # Adds batch dimension (1, 480, 764, 1)

        for batch in datagen.flow(numpy_df, batch_size=1):
            X_values.append(batch[0].astype(np.float32))  # Extract augmented data
            y_values.append(y_value)
            i += 1
            if i % 15 == 0:  # Generate 15 augmented samples
                break

# Resize each image to (128, 128)
X_resized = np.array([cv2.resize(x, (128, 128), interpolation=cv2.INTER_AREA) for x in X_values])

# Reshape to (128, 128, 1)
X_resized = X_resized.reshape(-1, 128, 128, 1)
X_values = X_resized / np.max(np.abs(X_resized))

# Save new dataset
np.savez("./data/augmented_data.npz", X_values=X_values, y_values=y_values)
