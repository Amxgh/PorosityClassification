import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

val_generator = datagen.flow_from_directory(
    "dataset", target_size=(128, 128), color_mode='grayscale',
    batch_size=32, class_mode='binary'
)

# Load the trained model
model = tf.keras.models.load_model("models/saved_model.h5")

test_loss, test_acc = model.evaluate(val_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
