import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint


class VGG:
    def __init__(self):
        try:
            self.image_size = (224, 224)
            self.train_path = 'C:\\Users\\Admin\\Desktop\\Python_Project\\Deep Learning Algo\\CNN\\images\\training_set'
            self.test_path = 'C:\\Users\\Admin\\Desktop\\Python_Project\\Deep Learning Algo\\CNN\\images\\test_set'

            self.training_data = ImageDataGenerator(rescale = 1.0/255.0,
                                                    shear_range = 0.3,
                                                    zoom_range = 0.3,
                                                    horizontal_flip = True,
                                                    )
            self.testing_data = ImageDataGenerator(rescale = 1.0/255.0)

            self.final_train = self.training_data.flow_from_directory(self.train_path,
                                                                      target_size=(224,224),
                                                                      batch_size=10,
                                                                      class_mode='binary')
            self.final_test = self.testing_data.flow_from_directory(self.test_path,
                                                                    target_size=(224,224),
                                                                    batch_size=5,
                                                                    class_mode='binary')
            # print(self.final_test.classes)
            # print(self.final_test.filenames)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def architecture(self):
        try:
            self.vgg16 = VGG16(input_shape=self.image_size+(3,), weights='imagenet', include_top=False)
            for layers in self.vgg16.layers:
                layers.trainable = False
            x = Flatten()(self.vgg16.output)
            x = Dense(128, activation = 'relu')(x)
            x = Dense(64, activation = 'relu')(x)
            output = Dense(1, activation = 'sigmoid')(x)
            self.model = Model(inputs=self.vgg16.input, outputs = output)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def metrics(self):
        try:
            checkpoint = ModelCheckpoint(
                filepath = 'best_weights.keras',
                monitor = 'val_loss',
                save_best_only = True,
                verbose = 1
            )
            self.model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])
            self.model.fit(self.final_train,
                           epochs=20,
                           validation_data = self.final_test,
                           callbacks = [checkpoint])
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

    def evaluate(self):
        try:
            self.model.load_weights('best_weights.keras')
            predictions = self.model.predict(self.final_test)
            prediction_labels = (predictions > 0.5).astype(int)

            filenames = self.final_test.filenames
            true_labels = self.final_test.classes

            print(f"Accuracy Score: {accuracy_score(true_labels, prediction_labels)}")
            print('-------------------------------------------------------------------------------')
            print(f"Classification Report: \n{classification_report(true_labels, prediction_labels)}")
            print('-------------------------------------------------------------------------------')
            print(f'Confusion Matrix: \n{confusion_matrix(true_labels, prediction_labels)}')
            print('-------------------------------------------------------------------------------')

            for i, (filename, pred, true) in enumerate(zip(filenames, prediction_labels, true_labels)):
                print(f"Image: {filename}, Prediction: {'Dog' if pred == 1 else 'Cat'}, Actual: {'Dog' if true == 1 else 'Cat'}\n")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')



if __name__ == "__main__":
    try:
        vgg16 = VGG()
        vgg16.architecture()
        vgg16.metrics()
        vgg16.evaluate()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'Error Type: {er_type} \n Error Msg: {er_msg} \n Error Line: {er_line.tb_lineno}')

