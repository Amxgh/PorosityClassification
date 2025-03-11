import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the dataset
dataset_path = "dataset/correct/"
augmented_path = "dataset/augmented/"

# Initialize the data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Function to check if the file is a valid CSV
def is_valid_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data is not None
    except:
        return False


# Loop through each file in the dataset
for filename in os.listdir(dataset_path):
    filepath = os.path.join(dataset_path, filename)

    # Only process CSV files
    if not filename.lower().endswith('.csv'):
        continue

    if is_valid_csv(filepath):
        data = pd.read_csv(filepath)

        # Assuming each CSV is a matrix of temperature values, normalize the data
        data = data.values  # Convert DataFrame to numpy array

        # Optionally reshape or preprocess the data (e.g., resize to fit model)
        # Here, we assume the matrix should be reshaped to (128, 128)
        data = np.resize(data, (128, 128))  # Resize to 128x128 if necessary

        # Expand dimensions to simulate batch size for data augmentation
        data = np.expand_dims(data, axis=-1)  # Add channel dimension
        data = np.expand_dims(data, axis=0)  # Add batch dimension

        # Generate and save augmented data
        i = 0
        for batch in datagen.flow(data, batch_size=1, save_to_dir=augmented_path,
                                  save_prefix='aug'):

            # Save the augmented data as a CSV file
            augmented_data = batch[0, :, :, 0]  # Get the augmented matrix
            augmented_filename = f"{augmented_path}aug_{i}_{filename}"
            np.savetxt(augmented_filename, augmented_data, delimiter=",")
            i += 1
            if i >= 5:  # Generate 5 augmented matrices per original data
                break
    else:
        print(f"Skipped non-CSV file: {filename}")

print("Data augmentation completed.")
