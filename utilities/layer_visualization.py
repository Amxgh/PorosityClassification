import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from IPython.display import display, Image as IPImage
import matplotlib.colors as mcolors
import pandas as pd
import cv2


# --- Load and preprocess CSV image ---
df = pd.read_csv("../data/healthy/norrmal_2025-04-01_19-17-43_3.csv", index_col=None, header=None)
# df = pd.read_csv("../data/unhealthy/Abnormal_2025-04-01_19-09-34_1.csv", index_col=None, header=None)
df = df.dropna(axis=1, how='any')
numpy_df = df.to_numpy()
numpy_df = cv2.resize(numpy_df, (128, 128), interpolation=cv2.INTER_AREA)

# Normalize and reshape
temp = (numpy_df - numpy_df.min()) / (numpy_df.max() - numpy_df.min())
sample = temp[None, ..., None].astype("float32")  # shape: (1,128,128,1)


# --- Load model and extract desired activation ---
model = tf.keras.models.load_model("../utilities/models/best_model_SGD_0.01")
conv_layers = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)]

# --- User choice: select specific layer and filter index ---
layer_name = "conv2d"      # ← replace with your layer name
filter_index = 4            # ← replace with desired filter index

# --- Build activation model ---
layer_dict = {l.name: l for l in conv_layers}
selected_layer = layer_dict[layer_name]
act_model = keras.Model(model.inputs, [selected_layer.output])
feature_map = act_model(sample)[0, :, :, filter_index].numpy()


# --- Color mapping ---
def colourise(gray, cmap):
    rgb = cmap(gray / 1.0)[..., :3]  # assumes normalized input
    return (rgb * 255).astype("uint8")

vmin = feature_map.min()
vmax = feature_map.max()
mean = feature_map.mean()
std = feature_map.std()

def norm(val): return (val - vmin) / (vmax - vmin)
def clamp(v): return np.clip(v, vmin, vmax)

# Define asymptotic colormap
color_points = [
    (0.00, "#000000"),  # black
    (0.60, "#0d0221"),  # near black
    (0.70, "#3c096c"),  # dark purple
    (0.75, "#f72585"),  # bright pink
    (0.85, "#fca311"),  # orange
    (0.92, "#ffee80"),  # soft yellow
    (0.98, "#ffffcc"),  # pale white
    (1.00, "#ffffff"),  # pure white
]

cmap = mcolors.LinearSegmentedColormap.from_list("asymptotic_purplehot", color_points)


# Standardize and clip the feature map
g = feature_map
g -= g.mean(); g /= (g.std() + 1e-5); g = np.clip(g*0.2 + 0.5, 0, 1)

# Colorize and save
colored_image = colourise(g, cmap)
filename = f"{layer_name}_filter{filter_index}_activation_normal.png"
cv2.imwrite(filename, colored_image)

# Display
display(IPImage(filename))
