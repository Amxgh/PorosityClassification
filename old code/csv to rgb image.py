

# Function to check if the file is a valid CSV
def is_valid_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data is not None
    except:
        return False

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
csv_file = "dataset_image3.csv"  # Change this to your CSV file path
data = pd.read_csv(csv_file, header=None)  # Assuming no header

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create a custom colormap
colors = [
    (0, 0, 0),    # Black for lowest temperatures
    (0, 0, 1),    # Blue at 1800°F
    (1, 0, 0),    # Red at 2750°F
    (1, 0, 0)     # Full Red for 3000+°F
]

# Define the range of temperatures
temperature_values = [1000, 1800, 2750, 3000]  # Corresponding temperature cutoffs

# Normalize the colormap
norm = mcolors.Normalize(vmin=1000, vmax=3000)  # Set min/max for scaling
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_heatmap", colors, N=256)

# # Generate synthetic data for testing
# data = np.random.uniform(1000, 3000, (100, 100))  # Replace this with actual temperature data
#
# # Plot the heatmap
# plt.figure(figsize=(8, 6))
# plt.imshow(data, cmap=custom_cmap, norm=norm)
# plt.colorbar(label="Temperature (°F)")
# plt.title("Custom Temperature Heatmap")
# plt.show()



# data = data / (np.max(np.abs(data)))
# Convert data to numpy array
temperature_array = data.to_numpy()

# Create the figure and axis
plt.figure(figsize=(10, 8))

# Display the heatmap
plt.imshow(temperature_array, cmap=custom_cmap, interpolation='nearest')

# Add colorbar for reference
plt.colorbar(label="Temperature")

# Display the image
plt.title("Temperature Heatmap")
plt.show()

