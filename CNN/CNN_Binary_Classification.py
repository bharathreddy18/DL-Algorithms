import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True
            )

            self.test_datagen = ImageDataGenerator(
                rescale = 1.0/255.0)

            labels = ['cats', 'dogs']
            self.train_final_data = self.train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(256,256),
                classes = labels,
                batch_size = 9,
                class_mode='binary'
            )

            self.test_final_data = self.test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(256, 256),
                classes=labels,
                batch_size=5,
                class_mode='binary'
            )
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def architecture(self):
        try:
            self.model = Sequential() # Initializing the model

            # Kernel Layers or Filter layers
            self.model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(256, 256, 3), padding='same'))
            self.model.add(MaxPool2D(pool_size=(2,2)))

            self.model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
            self.model.add(MaxPool2D(pool_size=(2, 2)))

            # self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
            # self.model.add(MaxPool2D(pool_size=(2, 2)))
            #
            # self.model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
            # self.model.add(MaxPool2D(pool_size=(2, 2)))

            self.model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
            self.model.add(MaxPool2D(pool_size=(2, 2)))

            self.model.add(Flatten())  # One Dimensional Array

            #Hidden Layers
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dense(16, activation='relu'))
            self.model.add(Dense(8, activation='relu'))

            # Output Layer
            self.model.add(Dense(1, activation='sigmoid'))

            #Model Summary
            self.model.summary()
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def metrics(self):
        try:
            self.model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

            self.model.fit(self.train_final_data,
                           validation_data = self.test_final_data,
                           epochs=5)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def evaluate(self):
        try:
            test_loss, test_accuracy = self.model.evaluate(self.test_final_data)
            print(f'Testing accuracy: {test_accuracy*100:.2f}%')
            print(f'Testing Loss: {test_loss}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def checking(self, path):
        try:
            img = cv2.imread(path)
            resized_img = cv2.resize(img, (256, 256))   # Resizing test image
            sc_test_img = resized_img / 255.0       # Scaling down test image
            final_test_img = np.expand_dims(sc_test_img, axis=0)
            if self.model.predict(final_test_img)[0][0] > 0.5:
                print('Dog')
            else:
                print('Cat')
            cv2.imshow('test image', resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')


if __name__ == "__main__":
    try:
        cnn = CNN()
        cnn.architecture()
        cnn.metrics()
        cnn.evaluate()
        cnn.checking("C:\\Users\\Admin\\Downloads\\download.jpeg")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')
