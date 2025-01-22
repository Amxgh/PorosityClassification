import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load a thermal image in grayscale
image_path = "image.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the image
plt.imshow(image, cmap='hot')
plt.title("Thermal Image")
plt.colorbar()
plt.show()

# Normalize pixel values to the range [0, 1]
normalized_image = image / 255.0

