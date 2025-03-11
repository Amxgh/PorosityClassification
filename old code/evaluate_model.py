import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Load trained autoencoder
autoencoder = tf.keras.models.load_model("models/autoencoder.h5")

# Path to test CSV files
test_path = "test_images"

threshold = 0.02  # Set a threshold for reconstruction error

for filename in os.listdir(test_path):
    filepath = os.path.join(test_path, filename)

    # Load CSV file
    if filename.lower().endswith('.csv'):
        data = pd.read_csv(filepath)
        data = data.values  # Convert DataFrame to numpy array

        # Reshape or preprocess the data as needed (e.g., resize to 128x128)
        data = np.resize(data, (128, 128))  # Resize matrix to 128x128
        data = np.expand_dims(data, axis=-1)  # Add channel dimension
        data = np.expand_dims(data, axis=0)  # Add batch dimension

        # Reconstruct the data with the autoencoder model
        reconstructed = autoencoder.predict(data)

        # Compute reconstruction error
        error = np.mean((data - reconstructed) ** 2)

        # Classify as acceptable or unacceptable
        if error > threshold:
            print(f"{filename}: ❌ Unacceptable (Error: {error:.5f})")
        else:
            print(f"{filename}: ✅ Acceptable (Error: {error:.5f})")
