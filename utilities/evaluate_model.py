import tensorflow as tf
import numpy as np
import cv2
import os

# Load trained autoencoder
autoencoder = tf.keras.models.load_model("models/autoencoder.h5")

# Path to test images (both acceptable and unacceptable)
test_path = "test_images"

threshold = 0.02  # Set a threshold for reconstruction error

for filename in os.listdir(test_path):
    filepath = os.path.join(test_path, filename)

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=[0, -1])  # Expand for model input

    # Reconstruct the image
    reconstructed = autoencoder.predict(image)

    # Compute reconstruction error
    error = np.mean((image - reconstructed) ** 2)

    # Classify as acceptable or unacceptable
    if error > threshold:
        print(f"{filename}: ❌ Unacceptable (Error: {error:.5f})")
    else:
        print(f"{filename}: ✅ Acceptable (Error: {error:.5f})")
