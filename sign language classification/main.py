import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data(filename):
    # You will need to write code that will read the file passed
    # into this function. The first line contains the column headers
    # so you should ignore it
    # Each successive line contains 785 comma separated values between 0 and 255
    # The first value is the label
    # The rest are the pixel values for that picture
    # The function will return 2 np.array types. One with all the labels
    # One with all the images
    #
    # Tips:
    # If you read a full line (as 'row') then row[0] has the label
    # and row[1:785] has the 784 pixel values
    # Take a look at np.array_split to turn the 784 pixels into 28x28
    # You are reading in strings, but need the values to be floats
    # Check out np.array().astype for a conversion
    with open(filename) as training_file:
        labels = []
        images = []
        lines = training_file.readlines()
        for line in lines:
            values = line.split(',')
            if values[0] == 'label':
                continue
            labels.append(values[0])
            images.append(np.array_split(np.array(values[1:785]), indices_or_sections=28))

    images = np.array(images).astype(dtype='float')
    labels = np.array(labels).astype(dtype='float')
    return images, labels


training_images, training_labels = get_data('sign_mnist_train.csv')
testing_images, testing_labels = get_data('sign_mnist_test.csv')

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

# In this section you will have to add another dimension to the data
# So, for example, if your array is (10000, 28, 28)
# You will need to make it (10000, 28, 28, 1)
# Hint: np.expand_dims

training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    horizontal_flip=True,
    zoom_range=0.2,
    height_shift_range=0.2,
    width_shift_range=0.2,
    fill_mode='nearest',
    rotation_range=40,
)

validation_datagen = ImageDataGenerator(rescale=1 / 255)

# Keep These
print(training_images.shape)
print(testing_images.shape)

# Their output should be:
# (27455, 28, 28, 1)
# (7172, 28, 28, 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=16, kernel_size=(3, 3), strides=1),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),


    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=26, activation='softmax'),

])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
    generator=train_datagen.flow(x=training_images,y=training_labels,batch_size=32,),
    epochs=15,
    steps_per_epoch=len(training_images) / 32,
    validation_data=validation_datagen.flow(x=testing_images,y=testing_labels,batch_size=32,),
    validation_steps=len(testing_images) / 32,
)
# Their output should be:
# (27455, 28, 28)
# (27455,)
# (7172, 28, 28)
# (7172,)
