# Machine Learning Model to Identify Faulty Welds

This repository contains a machine learning model that identifies faulty welds in thermal images. The model is trained
on a dataset of thermal images of welds. The dataset contains thermal images of welds and whether they have porosity or
not. The model is a convolutional neural network that is trained on the dataset to identify faulty welds. The model is
then evaluated on a test set and saved to a file.

## dataset_filtering.py

This script uses an extracted folder
of [this dataset]( https://www.sciencedirect.com/science/article/pii/S235234092300793X).
The dataset contains thermal images (csv files of the welds). The csv files have temperature of each "pixel" that it
sees.

1. The script opens thermal-porosity-table.csv and extracts the Y-values (whether it has porosity or not) for each of
   the frames.
2. The script opens each of csv files and appends the data to a numpy array.
3. Once all the images have been appended to the array, the data is normalized by dividing it by the maximum absolute
   value in the array. This normalizes the values between 0 and 1.
4. The script then saves the X and Y values into a npz (numpy zip) file called dataset.npz.

## ./utilities/train_model.py

This script uses the dataset.npz file to train a convolutional neural network to identify faulty welds. It then evalues
the model on the test set and then saves the model to ./models/model.

1. The script loads up dataset.npz and saves the X and Y values into X_values and Y_values respectively.
2. The X values are then reshaped to be (number of images, 200, 201)
3. The data (X_values and Y_values) is then split into X_train, X_test, y_train, y_test.
4. The training data (X_train and y_train) is then split into training and validation data (X_train, Y_train, X_val,
   Y_val).
5. Each of training data, validation data and test data are loaded into CSVDataGenerator objects. These objects are used
   to load the data into the model in batches.
6. Now, we start defining the model. The model is a convolutional neural network defined using tensorflow keras. It
   performs binary classification (whether the weld is faulty or not). The model is then compiled using the Adam
   optimizer and binary crossentropy loss. Following is the description for each of the layers:
    - **Input Layer**: Receives a grayscale image that is 200x201 pixels.
    - **First Hidden Layer** (Feature Extraction):
        - Applies 64 convolutional filters to extract more complex patterns from the image.
        - Batch normalization is applied to normalize the a values of the previous layer at each batch.
        - Relu activation is used.
    - **Second Hidden Layer** (Feature Extraction):
        - Applies 64 convolutional filters to extract the edges and other features from the image.
        - Batch normalization is applied to normalize the a values of the previous layer at each batch.
        - Relu activation is used.
        - Max pooling is applied to reduce the size of the image. This improves efficiency.
    - **Third Hidden Layer** (Feature Extraction):
        - Applies 32 filters to capture more features.
        - Batch normalization is applied to normalize the a values of the previous layer at each batch.
        - Relu activation is used.
    - **Fourth Hidden Layer** (Feature Extraction):
        - Applies 32 filters to capture more features.
        - Batch normalization is applied to normalize the a values of the previous layer at each batch.
        - Relu activation is used.
    - **Fifth Hidden Layer** (Feature Extraction):
        - Applies 16 filters to capture more features.
        - Batch normalization is applied to normalize the a values of the previous layer at each batch.
        - Relu activation is used.
    - **Sixth Hidden Layer** (Feature Extraction):
        - Applies 8 filters to capture more features.
        - Batch normalization is applied to normalize the a values of the previous layer at each batch.
        - Relu activation is used.
    - **Seventh Hidden Layer** (Flatten):
        - Flattens the output of the previous layer to be fed into the dense layer.
    - **Eighth Hidden Layer** (Dense):
        - Applies 128 units to capture the features from the previous layer.
        - Relu activation is used.
        - Has a dropout of 0.4 to prevent overfitting.
    - **Ninth Hidden Layer** (Dense):
        - Applies 64 units to capture the features from the previous layer.
        - Relu activation is used.
        - Has a dropout of 0.4 to prevent overfitting.

   _Hidden layers 8 and 9 are fully connected layers_

    - **Output Layer**:
        - Applies 1 unit to classify the image as faulty or not.
        - Sigmoid activation is used to output a probability.

7. The model is then compiled using the adam optimizer with a learning rate of 0.0001 and binary crossentropy loss.
8. The model is then fit using the training data and validation data. The model is trained for 20 epochs (Takes around
   10 minutes).
9. The model is then evaluated on the test data and the accuracy is printed.
10. The model is then saved to ./models/model.