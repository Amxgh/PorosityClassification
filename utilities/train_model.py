import numpy as np
import tensorflow.keras.regularizers as regularizers
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

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

X_values = X_values.reshape(1564, 200, 201)

X_train, X_test, Y_train, Y_test = train_test_split(X_values, Y_values, test_size=0.15, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1765, random_state=42)

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

input_img = Input(shape=(200, 201, 1))

# First hidden layer (feature extraction)
x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Second hidden layer (feature extraction)
x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Third hidden layer (feature extraction)
x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Fourth hidden layer (feature extraction)
x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Fifth hidden layer (feature extraction)
x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Sixth hidden layer (feature extraction)
x = Conv2D(8, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Seventh hidden layer (flatten)
x = Flatten()(x)

# Eighth hidden layer (fully connected)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.4)(x)

# Ninth hidden layer (fully connected)
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

# Train the model
model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_val, Y_val))
print("Autoencoder model created.")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Saving the model
model.save("models/model", save_format="tf")

print("Training Completed")
