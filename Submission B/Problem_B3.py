# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import activations, layers
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dropout, Flatten
from tensorflow.python.keras.layers.pooling import MaxPool2D, MaxPooling2D

def solution_B3():
    #data_url = 'https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/rps.zip'
    #urllib.request.urlretrieve(data_url, 'rps.zip')
    #local_file = 'rps.zip'
    #zip_ref = zipfile.ZipFile(local_file, 'r')
    #zip_ref.extractall('data/')
    #zip_ref.close()

    TRAINING_DIR = "data/rps/"
    VALIDATION_DIR = "data/rps/"
    training_datagen = ImageDataGenerator(rescale = 1./255)# YOUR CODE HERE

    train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size = 32,
                                                    class_mode = 'categorical', 
                                                    target_size = (150, 150))# YOUR CODE HERE

    validation_datagen = ImageDataGenerator(rescale = 1./255)# YOUR CODE HERE

    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                    batch_size = 32,
                                                    class_mode = 'categorical', 
                                                    target_size = (150, 150))

    model = tf.keras.models.Sequential([
        # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(300),
        tf.keras.layers.Dense(150),
        tf.keras.layers.Dense(75),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_generator, epochs=10, verbose=1, validation_data=validation_generator)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B3()
    model.save("model_B3.h5")

