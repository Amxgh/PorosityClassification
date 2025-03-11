import pandas as pd

# Load CSV file
csv_file = "dataset/augmented/aug_1_Frame_27.csv"  # Change this to your CSV file path
data = pd.read_csv(csv_file, header=None)  # Assuming no header

import matplotlib.pyplot as plt

temperature_array = data.to_numpy()

# Create the figure and axis
plt.figure(figsize=(10, 8))

# Display the heatmap
plt.imshow(temperature_array, cmap="rainbow", interpolation='nearest')

# Add colorbar for reference
plt.colorbar(label="Temperature")

# Display the image
plt.title("Temperature Heatmap")

plt.show()