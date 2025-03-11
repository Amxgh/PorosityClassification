import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.ndimage import zoom  # Used for resizing numerical data if needed

# from utilities.train_model import X_values

# Load the trained model
model = tf.keras.models.load_model("./utilities/models/march11")

def classify_csv(csv_path):
    # Load CSV file (assuming it's just raw temperature values)
    df = pd.read_csv(csv_path).iloc[:, 1:]  # Drops the index column
    data = df.to_numpy()
    # print(data)

    # Ensure correct shape (200, 201) - Rescale only if needed
    if data.shape != (200, 201):
        data = zoom(data, (200 / data.shape[0], 201 / data.shape[1]))  # Resize while keeping proportions

    # Normalize data (temperature range normalization)
    X_values = data / np.max(np.abs(data))
    # min_temp = np.min(data)
    # max_temp = np.max(data)
    # X_values = (data - min_temp) / (max_temp - min_temp + 1e-8)  # Normalize between 0 and 1
    # print("Processed Frame Data:", X_values.shape, X_values.min(), X_values.max())

    # Add necessary dimensions for model input
    X_values = np.expand_dims(X_values, axis=-1)  # Add channel dimension (grayscale)
    X_values = np.expand_dims(X_values, axis=0)  # Add batch dimension

    # Predict using the model
    prediction = model.predict(X_values)[0][0]
    print(prediction)
    # Return classification result
    return "Unhealthy" if prediction >= 0.5 else "Healthy"

# Example usage
print("Frame 6" + classify_csv("./dataset/all_data/Frame_6"))
print("Frame 7" + classify_csv("./dataset/all_data/Frame_7"))
print("Frame 8" + classify_csv("./dataset/all_data/Frame_8"))
print("Frame 9" + classify_csv("./dataset/all_data/Frame_9"))
print("Frame 10" + classify_csv("./dataset/all_data/Frame_10"))
print("Frame 11" + classify_csv("./dataset/all_data/Frame_11"))
print("Frame 27" + classify_csv("./dataset/all_data/Frame_27"))

