import os
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Use an appropriate backend (change 'TkAgg' if you need something else)
matplotlib.use('TkAgg')  # or 'Qt5Agg'

# Root directory containing CSV files
csv_file = "../data/healthy"  # Adjust this path as needed

# Gather all CSV files into a list
all_csv_files = []
for root, dirs, files in os.walk(csv_file, topdown=True):
    for file in files:
        if file.lower().endswith(".csv"):
            all_csv_files.append(os.path.join(root, file))

# Sort files if needed (optional)
# all_csv_files.sort()

# Batch size
batch_size = 5

# Create your colormap for consistent scaling across plots (optional)
colors = [
    (0, 0, 0),  # Black for lowest temperatures (1000°F)
    (0, 0, 1),  # Blue at 1800°F
    (1, 0, 0),  # Red at 2750°F
    (1, 0, 0)  # Full Red for 3000+°F
]
# Define the range of temperatures
temperature_values = [1000, 1800, 2750, 3000]  # temperature cutoffs
norm = mcolors.Normalize(vmin=1000, vmax=3000)
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_heatmap", colors, N=256)

# Calculate the number of batches
num_batches = math.ceil(len(all_csv_files) / batch_size)

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(all_csv_files))
    batch_files = all_csv_files[start_idx:end_idx]

    # Create a figure for this batch
    fig, axes = plt.subplots(1, len(batch_files), figsize=(5 * len(batch_files), 5))

    # If there is only one file in the batch, axes won't be a list. Ensure it’s iterable.
    if len(batch_files) == 1:
        axes = [axes]

    for ax, file_path in zip(axes, batch_files):
        data = pd.read_csv(file_path, header=None)
        temperature_array = data.to_numpy()

        # Plot each CSV as a heatmap
        img = ax.imshow(temperature_array, cmap='rainbow', interpolation='nearest')

        # Optional: If you want consistent color scaling across all subplots,
        # use the normalized colormap like:
        # img = ax.imshow(temperature_array, cmap=custom_cmap, norm=norm, interpolation='nearest')

        ax.set_title(f"File:\n{os.path.basename(file_path)}", fontsize=10)
        ax.axis('off')  # Hide the axis if desired
        # Colorbar for each subplot (optional, can be omitted or replaced with a single colorbar)
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
