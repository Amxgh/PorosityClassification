import time

import numpy as np
import tensorflow.keras.regularizers as regularizers
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

optimizers = {
    'Adam': Adam,
    'SGD': SGD,
    'RMSprop': RMSprop
}

learning_rates = [0.01, 0.001, 0.0005, 0.0001]

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

X_values = X_values.reshape(1564, 200, 201, 1)

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

model = Model(inputs=input_img, outputs=output)

results = {}
for opt_name, opt in optimizers.items():
    for lr in learning_rates:
        optimizer = opt(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

        model.summary()

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(f"models/best_model_{opt_name}_{lr}",
                                           save_best_only=True,
                                           save_format="tf")

        start_time = time.time()
        history = model.fit(X_train, Y_train, epochs=20,
                            validation_data=(X_val, Y_val),
                            callbacks=[early_stop, model_checkpoint])

        end_time = time.time()
        print("Autoencoder model created.")

        training_time = end_time - start_time

        predictions = model.predict(X_test)

        predicted_classes = (predictions > 0.5).astype(int).flatten()

        accuracy = np.mean(Y_test == predicted_classes)
        conf_matrix = confusion_matrix(Y_test, predicted_classes)

        print("Accuracy: ", accuracy)
        print("Confusion Matrix: ", conf_matrix)

        results[(opt_name, lr)] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'training_time': training_time
        }

        print("Training Completed")

print("Results: ", results)
np.savez("results", results=results)
print("Results saved to results.npz")
