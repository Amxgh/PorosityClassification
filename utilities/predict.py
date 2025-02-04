import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("models/saved_model.h5")

def classify_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=(0, -1))

    prediction = model.predict(image)[0][0]
    return "Healthy" if prediction >= 0.5 else "Unhealthy"

# Example usage
print(classify_image("dataset/test/sample.jpg"))
