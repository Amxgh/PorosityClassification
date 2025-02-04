import tensorflow as tf
import numpy as np
import pandas as pd

model = tf.keras.models.load_model("models/saved_model.h5")

def classify_csv(csv_path):
    # Load CSV file
    data = pd.read_csv(csv_path)
    data = data.values  # Convert DataFrame to numpy array

    # Reshape the data (e.g., to 128x128)
    data = np.resize(data, (128, 128))  # Resize matrix to 128x128
    data = np.expand_dims(data, axis=-1)  # Add channel dimension
    data = np.expand_dims(data, axis=0)  # Add batch dimension

    # Predict using the model
    prediction = model.predict(data)[0][0]
    return "Healthy" if prediction >= 0.5 else "Unhealthy"

# Example usage
print(classify_csv("dataset/test/sample.csv"))
