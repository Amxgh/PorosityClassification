import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence


from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy

csv_dir = 'dataset/augmented/'


def load_csv_data(csv_dir):
    data = []
    for file_name in os.listdir(csv_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(csv_dir, file_name)
            df = pd.read_csv(file_path, header=None)  # Read CSV without headers
            data.append(df.values)  # Convert the DataFrame to a NumPy array
    return np.array(data)


# Load full data
csv_data = load_csv_data(csv_dir)
csv_data = csv_data.reshape((csv_data.shape[0], 128, 128, 1))
csv_data = csv_data / 255.0
csv_data = np.clip(csv_data, 0, 1)

# Split into training+validation and test (e.g., 85% train_val, 15% test)
train_val_data, test_data = train_test_split(csv_data, test_size=0.15, random_state=42)

# Further split train_val_data into training and validation (e.g., ~82% training, ~18% validation)
train_data, val_data = train_test_split(train_val_data, test_size=0.1765, random_state=42)

print("Data loaded and split into training, validation, and test sets.")



class CSVDataGenerator(Sequence):
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data[batch_indices]
        return batch_data, batch_data

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# Create generators
train_data_generator = CSVDataGenerator(train_data, batch_size=32)
val_data_generator = CSVDataGenerator(val_data, batch_size=32)
test_data_generator = CSVDataGenerator(test_data, batch_size=32, shuffle=False)




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




input_img = Input(shape=(128, 128, 1))

# Encoder
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_img)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)  # Dropout to reduce overfitting

x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)

# Bottleneck
x = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)

# Decoder
x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss=BinaryCrossentropy(from_logits=False), metrics=['accuracy'])

# Early stopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Autoencoder model created.")

# Train the model
autoencoder.fit(train_data_generator,
                epochs=5,
                validation_data=val_data_generator)

test_loss, test_accuracy = autoencoder.evaluate(test_data_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Saving the model
autoencoder.save("models/autoencoder-updated", save_format="tf")

print("Training Completed")
