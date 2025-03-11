import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.utils import Sequence


from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy


dataset_path = "../dataset/dataset.npz"
try:
    # Load the npz file
    data = np.load(dataset_path, allow_pickle=True)
except FileNotFoundError:
    print(f"Error: '{dataset_path}' not found. Please check the file path.")
    exit()
except KeyError as e:
    print(f"Error: The expected key {e} is not found in the npz file. Check available keys with data.files")
    exit()

#
#
# csv_dir = '../dataset/augmented/'
#
#
# def load_csv_data(csv_dir):
#     data = []
#     for file_name in os.listdir(csv_dir):
#         if file_name.endswith('.csv'):
#             file_path = os.path.join(csv_dir, file_name)
#             df = pd.read_csv(file_path, header=None)  # Read CSV without headers
#             data.append(df.values)  # Convert the DataFrame to a NumPy array
#     return np.array(data)

X_values = data['X_values']
Y_values = data['Y_values']

X_values = X_values.reshape(1564, 200, 201)

# print(X_values.shape)
# print(Y_values.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X_values, Y_values, test_size=0.15, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1765, random_state=42)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

print("Data loaded and split into training, validation, and test sets.")

class CSVDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        return batch_X, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)



train_data_generator = CSVDataGenerator(X_train, Y_train, batch_size=32)
val_data_generator = CSVDataGenerator(X_val, Y_val, batch_size=32)
test_data_generator = CSVDataGenerator(X_test, Y_test, batch_size=32, shuffle=False)

print("Data Generators created.")



from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# input_img = Input(shape=(128, 128, 1))
#
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
#
# # Decoder
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#
# # Create model
# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adam', loss='mse')
#
#
#
#
# input_img = Input(shape=(200, 201, 1))
#
# # Encoder
# x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_img)
# x = BatchNormalization()(x)
# x = Dropout(0.2)(x)  # Dropout to reduce overfitting
#
# x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_regularizer=l2(0.001))(x)
# x = BatchNormalization()(x)
# x = Dropout(0.2)(x)
#
# x = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_regularizer=l2(0.001))(x)
# x = BatchNormalization()(x)
#
# # Bottleneck
# x = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
#
# # Decoder
# x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
# x = UpSampling2D((2, 2))(x)
#
# x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
# x = UpSampling2D((2, 2))(x)
#
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#
# autoencoder = Model(input_img, decoded)
#
# autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss=BinaryCrossentropy(from_logits=False), metrics=['accuracy'])
#
# # Early stopping callback
# # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Input
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.regularizers as regularizers

input_img = Input(shape=(200, 201, 1))

# Feature Extraction (Encoder)
x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)  # Pooling instead of stride

x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
# x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(8, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Flatten for classification
x = Flatten()(x)

# Fully Connected Layer
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
# x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.4)(x)

# Output Layer (Binary Classification)
output = Dense(1, activation='sigmoid')(x)

# Create Model
model = Model(inputs=input_img, outputs=output)


# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Model Summary
model.summary()

print("Autoencoder model created.")

# Train the model
# Assuming X_train contains reshaped thermal images and y_train has labels (0 or 1)
model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_val, Y_val))

test_loss, test_accuracy = model.evaluate(test_data_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Saving the model
model.save("models/march11", save_format="tf")

print("Training Completed")

