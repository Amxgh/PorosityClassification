import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the dataset
dataset_path = "dataset/correct"
augmented_path = "dataset/augmented"


datagen = ImageDataGenerator(
    rotation_range=20,  # Random rotations
    width_shift_range=0.1,  # Horizontal shifts
    height_shift_range=0.1,  # Vertical shifts
    zoom_range=0.2,  # Zoom in/out
    horizontal_flip=True,  # Horizontal flips
    fill_mode='nearest'  # Fill mode for new pixels
)

# Loop through each image in the dataset
for filename in os.listdir(dataset_path):
    filepath = os.path.join(dataset_path, filename)
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue


    image = cv2.resize(image, (128, 128))  # Resize image

    image = np.expand_dims(image, axis=-1)  # Make the image grayscale

    image = np.expand_dims(image, axis=0)  # clarify that each input is just one image

    # Generate and save augmented images
    i = 0
    for batch in datagen.flow(image, batch_size=1, save_to_dir=augmented_path,
                              save_prefix='aug', save_format='jpg'):
        i += 1
        if i >= 5:  # Generate 5 augmented images per original image
            break

print("Data augmentation completed.")
