import itertools
import time
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization,
                                     Activation, Dropout, GlobalAveragePooling2D, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight

dataset_path = "../data/augmented_data.npz"
# dataset_path = "../resized_data.npz"
try:
    data = np.load(dataset_path, allow_pickle=True)
except FileNotFoundError:
    print(f"Error: '{dataset_path}' not found. Please check the file path.")
    exit()
except KeyError as e:
    print(f"Error: The expected key {e} is not found in the npz file. Check available keys with data.files")
    exit()

X_values = data['X_values']
Y_values = data['y_values']

X_values = X_values.astype(np.float32)


for i in range(len(X_values)):
    std = X_values[i].std()
    if std < 1e-10:
        std = 1.0
    X_values[i] = (X_values[i] - X_values[i].mean()) / std

# X_values = X_values / np.max(np.abs(X_values))

# Ensure the right shape
X_values = X_values.reshape(-1, 128, 128, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X_values, Y_values,
                                                    test_size=0.15,
                                                    random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                  test_size=0.1765,
                                                  random_state=42)



classes = np.unique(Y_train)

# Compute class weights for each class
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=Y_train
)


# Convert to a dictionary {class_label: weight_value, ...}
class_weight_dict = dict(zip(classes, weights))

optimizers = {
    # 'RMSprop': RMSprop,
    # 'Adam': Adam,
    'SGD': SGD
}

# learning_rates = [0.01, 0.001, 0.0005, 0.0001]
learning_rates= [0.01]

results = {}
# for opt_name, opt in optimizers.items():
#     for lr in learning_rates:
pool_size = [(2, 2), (3, 3)]
combinations = list(itertools.product(pool_size, repeat=3))

for i, (p1, p2, p3) in enumerate(combinations):
    tf.keras.backend.clear_session()

    input_img = Input(shape=(128, 128, 1))

    x = Conv2D(32, (3,3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=p1)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=p2)(x)
    x = Dropout(0.1)(x)

    x = Conv2D(128, (5,5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=p3)(x)
    x = Dropout(0.2)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Removed the custom bias init
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_img, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),  # Using a fixed learning rate for simplicity
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")]
    )
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("models/best_model_simplified",
                                       save_best_only=True,
                                       save_format="tf")
    start_time = time.time()
    history = model.fit(
        X_train, Y_train,
        epochs=20,
        validation_data=(X_val, Y_val),
        callbacks=[early_stop, model_checkpoint],
        class_weight=class_weight_dict
    )
    end_time = time.time()

    val_preds = model.predict(X_val)
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        val_pred_classes = (val_preds > t).astype(int)
        current_f1 = f1_score(Y_val, val_pred_classes)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    print(f"Optimal threshold on validation set: {best_threshold:.2f} with F1 score: {best_f1:.4f}")

    test_preds = model.predict(X_test)
    predicted_classes = (test_preds > best_threshold).astype(int).flatten()

    accuracy = accuracy_score(Y_test, predicted_classes)
    conf_matrix = confusion_matrix(Y_test, predicted_classes)
    precision = precision_score(Y_test, predicted_classes)
    recall = recall_score(Y_test, predicted_classes)

    print("Test Accuracy:", accuracy)
    print("Test Confusion Matrix:\n", conf_matrix)
    print("Test Precision:", precision)
    print("Test Recall:", recall)

    print("Train label distribution:", np.bincount(Y_train.astype(int)))
    print("Validation label distribution:", np.bincount(Y_val.astype(int)))
    print("Test label distribution:", np.bincount(Y_test.astype(int)))

    results[(p1, p2, p3)] = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'time': end_time - start_time,
    }

    model.save(f"./utilities/models/kernels")
    print("Training Completed")

print("Results: ", results)
np.savez("results-finding-kernelsize", results=results)
print("Results saved to results.npz")
