import tensorflow as tf
import numpy as np
from keras.src.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, sigmoid
import os
import sys


class CNN:
    def __init__(self):
        try:
            self.train_dir = 'C:\\Users\\Admin\\Desktop\\Python_Project\\Deep Learning Algo\\CNN\\images\\training_set'
            self.test_dir = 'C:\\Users\\Admin\\Desktop\\Python_Project\\Deep Learning Algo\\CNN\\images\\test_set'

            # Data Augmentation for training
            self.train_datagen = ImageDataGenerator(
                rescale = 1.0/255.0,
                rotation_range = 20,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True
            )

            self.test_datagen = ImageDataGenerator(rescale = 1.0/255.0)

            # Generate Data batches
            self.train_generator = self.train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(150, 150),
                batch_size = 50,
                class_mode='binary'
            )

            self.test_generator = self.test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(150, 150),
                batch_size=50,
                class_mode='binary'
            )
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def architecture(self):
        try:
            self.model = Sequential()

            # self.model.add(Conv2D(128, kernel_size=(3,3), activation=relu, input_shape=(150, 150, 3)))
            # self.model.add(MaxPooling2D(pool_size=(2,2)))

            self.model.add(Conv2D(64, kernel_size=(5,5), activation=relu, input_shape=(150, 150, 3)))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(BatchNormalization())

            self.model.add(Conv2D(32, kernel_size=(5, 5), activation=relu))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(BatchNormalization())

            self.model.add(Conv2D(16, kernel_size=(5, 5), activation=relu))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(BatchNormalization())

            self.model.add(Conv2D(8, kernel_size=(5, 5), activation=relu))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(BatchNormalization())

            self.model.add(Flatten())

            self.model.add(Dense(128, activation=relu))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(1, activation=sigmoid))
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def metrics(self):
        try:
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            self.model.fit(self.train_generator,
                           validation_data = self.test_generator,
                           epochs=10)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def evaluate(self):
        try:
            test_loss, test_accuracy = self.model.evaluate(self.test_generator)
            print(f'Testing accuracy: {test_accuracy*100:.2f}%')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')


if __name__ == "__main__":
    try:
        cnn = CNN()
        cnn.architecture()
        cnn.metrics()
        cnn.evaluate()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')
