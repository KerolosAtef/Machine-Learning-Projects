import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, BatchNormalization
import shutil
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from shutil import copyfile
from os import getcwd

from PIL import Image
import numpy as np
from skimage import transform

base_dir = "D:\\Work\\Machine learning\\Datasets\\dogs-vs-cats"
training_dir = os.path.join(base_dir, 'training')
testing_dir = os.path.join(base_dir, 'testing')
source_cats_dir = os.path.join(base_dir, 'All dataset\\cats')
source_dogs_dir = os.path.join(base_dir, 'All dataset\\dogs')
training_cats_dir = os.path.join(training_dir, "cats")
training_dogs_dir = os.path.join(training_dir, "dogs")
testing_cats_dir = os.path.join(testing_dir, "cats")
testing_dogs_dir = os.path.join(testing_dir, "dogs")


def start():
    try:
        os.mkdir(training_dir)
        os.mkdir(testing_dir)

        os.mkdir(training_cats_dir)
        os.mkdir(training_dogs_dir)

        os.mkdir(testing_cats_dir)
        os.mkdir(testing_dogs_dir)

    except OSError:
        pass


def split_dataset(source, train, test, train_ratio):
    number_of_training_files = int(len(os.listdir(source)) * train_ratio)
    # number_of_testing_files = len(os.listdir(source)) - number_of_training_files
    count = 0
    for file in os.listdir(source):
        file_path = os.path.join(source, file)
        if os.path.getsize(file_path) <= 0:
            continue
        if count < number_of_training_files:
            copyfile(file_path, os.path.join(train, file))
        else:
            copyfile(file_path, os.path.join(test, file))

        count += 1

def load_image_from_path(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (150, 150, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def prepare_dataset():
    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.1,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=.2,
    )
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(150, 150),
        class_mode='binary',
        batch_size=32,
    )
    validation_datagen = ImageDataGenerator(rescale=1 / 255)
    validation_generator = validation_datagen.flow_from_directory(
        testing_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    return [train_generator, validation_generator]


def prepare_model(generators):
    train_generator = generators[0]
    validation_generator = generators[1]
    model = tf.keras.models.Sequential(layers=[
        Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='valid', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=2),

        Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid'),
        MaxPooling2D(pool_size=(2, 2), strides=2),

        Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='valid'),
        MaxPooling2D(pool_size=(2, 2), strides=2),

        Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='valid'),
        MaxPooling2D(pool_size=(2, 2), strides=2),

        Flatten(),
        Dense(units=512, activation='relu', use_bias=True, ),
        Dense(units=1, activation='sigmoid', use_bias=True, ),

    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(
    #     train_generator,
    #     epochs=15,
    #     validation_data=validation_generator
    # )
    # model.evaluate(validation_generator, )
    model.summary()
    return model


def use_transfer_learning_from_inception():
    pretrained_model = InceptionV3(input_shape=(150, 150, 3), weights=None, include_top=False)
    pretrained_weights_path = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    pretrained_model.load_weights(pretrained_weights_path)
    for layer in pretrained_model.layers:
        layer.trainable = False

    pretrained_model.summary()
    pretrained_output = pretrained_model.get_layer('mixed10').output
    x = Flatten()(pretrained_output)
    x = Dense(units=1024, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=1, activation='sigmoid')(x)
    model = tf.keras.models.Model(pretrained_model.input, x)
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # start()
    # split_dataset(source_cats_dir,training_cats_dir,testing_cats_dir,0.9)
    # split_dataset(source_dogs_dir,training_dogs_dir,testing_dogs_dir,0.9)

    # # train model
    # model = prepare_model(prepare_dataset())
    # # save model
    # model.save("Model with data augmentation 200 epoch.h5")
    #
    #
    # # load trained model
    # trained_model = tf.keras.models.load_model("Model with data augmentation.h5")
    # trained_model.summary()

    # Use transfer learning
    # train_generator, validation_generator = prepare_dataset()
    # new_inception_model = use_transfer_learning_from_inception()
    # new_inception_model.fit(
    #     train_generator,
    #     validation_data=validation_generator,
    #     epochs=20,
    #     verbose=2,
    # )
    # new_inception_model.save("new_inception_model_mixed10.h5")

    # load trained model
    trained_model = tf.keras.models.load_model("new_inception_model_mixed10.h5")
    # trained_model.summary()

    # # prediction part
    # test_path = "D:\\Work\\Machine learning\\Datasets\\dogs-vs-cats\\test1"
    # with open("inception mixed 10.csv", "w", encoding="utf-8") as fout:
    #     fout.write("id,label\n")
    #     for i in range(1,12501):
    #         m = trained_model.predict(load_image_from_path(os.path.join(test_path, str(i)+'.jpg')))
    #         fout.write(str(i)+','+str(m[0][0])+"\n")
    #         # if m[0] >= 0.5:
    #         #     print(i,'\tDog')
    #         #     # fout.write(str(i)+',1\n')
    #         # else:
    #         #     print(i,'\tCat')
    #         #     # fout.write(str(i)+',0\n')

    #special prediction

    mTestPath="my tests"
    for test in os.listdir(mTestPath):
        img=load_image_from_path(os.path.join(mTestPath,test))
        res =trained_model.predict(img)[0][0]
        print(test ,"\t",res)