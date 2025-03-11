import csv
import os
import numpy as np
import pandas as pd


dataset_path = "../dataset/dataset.npz"
try:
    data = np.load(dataset_path, allow_pickle=True)
except FileNotFoundError:
    print(f"Error: '{dataset_path}' not found. Please check the file path.")
    exit()
except KeyError as e:
    print(f"Error: The expected key {e} is not found in the npz file. Check available keys with data.files")
    exit()

X_values = data['X_values']
Y_values = data['Y_values']

# print(X_values.shape)
print("Number of frames:",Y_values.shape[0])
print("Number of unhealthy frames:",np.sum(Y_values))
print("Number of healthy frames:",Y_values.shape[0]-np.sum(Y_values))

X_values = X_values.reshape(1564, 200, 201)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(50, 10))
for i in range(3):
    ax = axes[i]
    ax.imshow(X_values[i], cmap="rainbow", interpolation='nearest')
    ax.set_title(f"Frame {i+1}")
    ax.set_xlabel("Temperature")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(X_values[563], cmap="rainbow", interpolation='nearest')
axes[0].set_title(f"Frame 564 - Unhealthy")
axes[0].set_xlabel("Temperature")

axes[1].imshow(X_values[90], cmap="rainbow", interpolation='nearest')
axes[1].set_title(f"Frame 91 - Healthy")
axes[1].set_xlabel("Temperature")

plt.show()


fig, axes = plt.subplots(1, 5, figsize=(10, 10))
unhealthy_frames = [215, 316, 417, 483, 509]
for i in range(5):
    ax = axes[i]
    ax.imshow(X_values[unhealthy_frames[i]], cmap="rainbow", interpolation='nearest')
    ax.set_title(f"Frame {unhealthy_frames[i]+1}")
    ax.set_xlabel("Temperature")
plt.show()

fig, axes = plt.subplots(1, 5, figsize=(10, 10))
healthy_frames = [642, 766, 912, 1226, 1361]
for i in range(5):
    ax = axes[i]
    ax.imshow(X_values[healthy_frames[i]], cmap="rainbow", interpolation='nearest')
    ax.set_title(f"Frame {healthy_frames[i]+1}")
    ax.set_xlabel("Temperature")
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(X_values[1293], cmap="rainbow", interpolation='nearest')
axes[0].set_title(f"Frame 1293 - Unhealthy - Middle")
axes[0].set_xlabel("Temperature")

axes[1].imshow(X_values[9], cmap="rainbow", interpolation='nearest')
axes[1].set_title(f"Frame 9 - Unhealthy - Beginning")
axes[1].set_xlabel("Temperature")

plt.show()

