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

# Resize the image to 128x128
resized_image = cv2.resize(normalized_image, (128, 128))

# ! DATA AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=20,       # Random rotation
    width_shift_range=0.1,   # Horizontal shift
    height_shift_range=0.1,  # Vertical shift
    zoom_range=0.2,          # Zoom in/out
    horizontal_flip=True,    # Flip horizontally
    fill_mode='nearest'      # Fill in missing pixels
)

