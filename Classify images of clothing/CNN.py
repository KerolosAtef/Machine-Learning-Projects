from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import keras.layers as ksl
from keras.models import Model

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


class ClothingClassification:
    def __init__(self):
        # Definig Global Variables
        self.fashion_mnist = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.class_names = None
        self.model = None

    def gettingDataset(self):
        self.fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.fashion_mnist.load_data()
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def applyNormalization(self):
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def showImage(self, idx):
        self.train_images = self.train_images.reshape((60000, 28, 28))
        self.test_images = self.test_images.reshape((10000, 28, 28))
        plt.figure()
        plt.imshow(self.test_images[idx])
        # plt.colorbar()
        plt.grid(False)
        plt.show()

    def showPortionOfImages(self, idx):
        plt.figure(figsize=(10, 10))
        for i in range(idx):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[self.train_labels[i]])
        plt.show()

    def makeNNModel(self):
        self.train_images = self.train_images.reshape((60000, 28, 28))
        self.test_images = self.test_images.reshape((10000, 28, 28))
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(500, activation='relu'),
            keras.layers.Dense(320, activation='relu'),
            keras.layers.Dense(200, activation='tanh'),
            keras.layers.Dense(100, activation='sigmoid'),
            keras.layers.Dense(10, activation='softmax')
        ])
        # keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.model.compile(optimizer='Adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def CNNModel(self):
        self.train_images = self.train_images.reshape((60000, 28, 28, 1))
        self.test_images = self.test_images.reshape((10000, 28, 28, 1))
        inp = ksl.Input((28, 28, 1))
        x = ksl.ZeroPadding2D((3, 3))(inp)
        x = ksl.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), name='conv0', kernel_initializer='he_uniform',
                       padding='same')(x)
        x = ksl.BatchNormalization(axis=3, name='bn0')(x)
        x = ksl.Activation('relu')(x)
        x = ksl.MaxPooling2D((2, 2), name='max_pool')(x)

        x = ksl.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), name='conv1', kernel_initializer='he_uniform',
                       padding='valid')(x)
        x = ksl.BatchNormalization(axis=3, name='bn1')(x)
        x = ksl.Activation('relu')(x)
        x = ksl.MaxPooling2D((2, 2), name='max_pool2')(x)

        x = ksl.Flatten()(x)
        x = ksl.Dense(50, activation='relu', name='fc1')(x)
        # x = ksl.Dense(25, activation='tanh', name='fc2')(x)
        x = ksl.Dense(10, activation='softmax', name='out')(x)
        self.model = Model(inputs=inp, outputs=x, name='CNN model')
        self.model.compile(optimizer='Adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    # evaluate a model using k-fold cross-validation
    def evaluate_model(self, dataX, dataY, n_folds=5):
        scores, histories = list(), list()
        # prepare cross validation
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        # enumerate splits
        for train_ix, test_ix in kfold.split(dataX):
            # define model
            # self.CNNModel()
            model = self.model
            # select rows for train and test
            trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
            # fit model
            history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
            # evaluate model
            _, acc = model.evaluate(testX, testY, verbose=0)
            print('> %.3f' % (acc * 100.0))
            # append scores
            scores.append(acc)
            histories.append(history)
        return scores, histories

    def trainModel(self):
        self.model.fit(self.train_images, self.train_labels, epochs=20, batch_size=40)

    def testModel(self):
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=0)
        print('\nTest accuracy:', test_acc)


if __name__ == '__main__':
    classifer = ClothingClassification()
    classifer.gettingDataset()
    classifer.applyNormalization()
    # classifer.showPortionOfImages(25)
    input = input("Do you want NN model or CNN model")
    if input == "NN":
        classifer.makeNNModel()
    elif input == "CNN":
        classifer.CNNModel()

    else:
        print("Error input")
        exit()
    classifer.trainModel()
    classifer.testModel()

    #To use K fold use this row 
    # classifer.evaluate_model(classifer.train_images, classifer.train_labels)

    predictions = classifer.model.predict(classifer.test_images)

    print(predictions[150])
    classifer.showImage(150)
    classifer.model.summary()
