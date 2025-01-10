import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import relu, softmax
import os
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class CNN:
    def __init__(self):
        try:
            self.train_dir = 'C:\\Users\\Admin\\Desktop\\Python_Project\\Deep Learning Algo\\CNN\\images\\Training_Data'
            self.test_dir = 'C:\\Users\\Admin\\Desktop\\Python_Project\\Deep Learning Algo\\CNN\\images\\Testing_Data'

            # Data Augmentation for training
            self.train_datagen = ImageDataGenerator(
                rescale = 1.0/255.0,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True
            )

            self.test_datagen = ImageDataGenerator(
                rescale = 1.0/255.0)

            self.labels = ['covid', 'normal', 'pnemonia']
            self.train_final_data = self.train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(256,256),
                classes = self.labels,
                batch_size = 10,
                class_mode='categorical'
            )

            self.test_final_data = self.test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(256, 256),
                classes=self.labels,
                batch_size=5,
                class_mode='categorical'
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
            self.model.add(Dense(3, activation='softmax'))

            #Model Summary
            self.model.summary()
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def metrics(self):
        try:
            self.model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

            self.model.fit(self.train_final_data,
                           validation_data = self.test_final_data,
                           epochs=5)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def evaluate(self):
        try:
            self.f = []
            test_loss, test_accuracy = self.model.evaluate(self.test_final_data)
            print(f'Testing accuracy: {test_accuracy*100:.2f}%')
            print(f'Testing Loss: {test_loss}')
            self.test_outcomes = self.model.predict(self.test_final_data)
            for i in self.test_outcomes:
                if self.labels[np.argmax(i)] == self.labels[0]:
                    self.f.append(0)
                elif self.labels[np.argmax(i)] == self.labels[1]:
                    self.f.append(1)
                else:
                    self.f.append(2)
            self.actual_points = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
            print(f'Accuracy Score: {accuracy_score(self.actual_points, self.f)}')
            print('--------------------------------------------------------------')
            print(f'Confusion Matrix: {confusion_matrix(self.actual_points, self.f)}')
            print('--------------------------------------------------------------')
            print(f'Classification Report: {classification_report(self.actual_points, self.f)}')
            print('--------------------------------------------------------------')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def checking(self, path):
        try:
            img = cv2.imread(path)
            resized_img = cv2.resize(img, (256, 256))   # Resizing test image
            sc_test_img = resized_img / 255.0       # Scaling down 0 to 1
            final_test_img = np.expand_dims(sc_test_img, axis=0)
            output = self.model.predict(final_test_img)
            print(self.labels[np.argmax(output)])
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
        cnn.checking("C:\\Users\\Admin\\Desktop\\Python_Project\\Deep Learning Algo\\CNN\\images\\Testing_Data\\covid\\Copy of COVID19(83).jpg")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')
