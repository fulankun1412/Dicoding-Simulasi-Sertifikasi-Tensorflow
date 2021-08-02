# =====================================================================================
# PROBLEM A2 
#
# Build a Neural Network Model for Horse or Human Dataset.
# The test will expect it to classify binary classes. 
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy and validation_accuracy > 83%
# ======================================================================================

import urllib.request
import zipfile
from numpy.lib.financial import rate
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.backend import dropout


def solution_A2():
    #data_url_1 = 'https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/horse-or-human.zip'
    #urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    #local_file = 'horse-or-human.zip'
    #zip_ref = zipfile.ZipFile(local_file, 'r')
    #zip_ref.extractall('data/horse-or-human')

    #data_url_2 = 'https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/validation-horse-or-human.zip'
    #urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    #local_file = 'validation-horse-or-human.zip'
    #zip_ref = zipfile.ZipFile(local_file, 'r')
    #zip_ref.extractall('data/validation-horse-or-human')
    #zip_ref.close()


    TRAINING_DIR = 'data/horse-or-human'
    VALIDATION_DIR = 'data/validation-horse-or-human'

    # YOUR CODE HERE
    train_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR, 
        target_size=(150, 150), 
        batch_size=128,
        class_mode='binary')

    test_datagen = ImageDataGenerator(rescale = 1./255 )

    validation_generator =  test_datagen.flow_from_directory(VALIDATION_DIR,
                                                          batch_size  = 32,
                                                          class_mode  = 'binary', 
                                                          target_size = (150, 150)) 

    model = tf.keras.models.Sequential([
        # YOUR CODE HERE, end with a Neuron Dense, activated by sigmoid
                tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    from tensorflow.keras.optimizers import Adam

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0003), metrics=['accuracy'])
    model.fit(train_generator,steps_per_epoch=8,  epochs=100, verbose=1,  validation_data = validation_generator, validation_steps= 8 )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A2()
    model.save("model_A2.h5")