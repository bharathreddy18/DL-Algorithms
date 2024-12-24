import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.activations import relu, softmax
from sklearn.metrics import classification_report, confusion_matrix


class ANN_Multilabel:
    def __init__(self):
        try:
            p = datasets.load_iris()
            self.df = pd.DataFrame(data = p.data, columns = p.feature_names)
            self.df['Species'] = p.target   # 0 - iris-setosa, 1 - iris-versicolor, 2 - iris-virginica
            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42) # Converts single column of output into 3 different columns because we gave 3 output layers in activation function.
            self.y_train_p = tensorflow.keras.utils.to_categorical(self.y_train, num_classes = 3)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Message: {er_msg} \n Error Traceback: {er_line.tb_lineno}')

    def architecture(self):
        try:
            self.model = Sequential()
            self.model.add(Dense(units=128, kernel_initializer='he_uniform', activation=relu, input_dim=self.X_train.shape[1]))
            self.model.add(Dense(units=64, kernel_initializer='he_uniform', activation=relu))
            self.model.add(Dense(units=32, kernel_initializer='he_uniform', activation=relu))
            self.model.add(Dense(units=16, kernel_initializer='he_uniform', activation=relu))
            self.model.add(Dense(units=8, kernel_initializer='he_uniform', activation=relu))
            self.model.add(Dense(units=3, kernel_initializer='glorot_uniform', activation=softmax))
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Message: {er_msg} \n Error Traceback: {er_line.tb_lineno}')

    def metrics(self):
        try:
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.fit(self.X_train, self.y_train_p, batch_size = 5, validation_split = 0.2, epochs = 50)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Message: {er_msg} \n Error Traceback: {er_line.tb_lineno}')

    def visualization(self):
        try:
            plt.figure(figsize=(9,5))
            plt.subplot(1,2,1)
            plt.xlabel('Epochs')
            plt.plot(np.arange(1,51), self.model.history.history['accuracy'], color = 'red', label = 'Train Accuracy')
            plt.plot(np.arange(1,51), self.model.history.history['loss'], color='blue', label = 'Train Loss')
            plt.legend()
            plt.subplot(1,2,2)
            plt.xlabel('Epochs')
            plt.plot(np.arange(1, 51), self.model.history.history['val_accuracy'], color='red', label='Val Accuracy')
            plt.plot(np.arange(1, 51), self.model.history.history['val_loss'], color='blue', label='Val Loss')
            plt.legend()
            plt.show()
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Message: {er_msg} \n Error Traceback: {er_line.tb_lineno}')

    def testing(self):
        try:
            self.d = []
            self.predictions = self.model.predict(self.X_test)
            self.class_labels = np.argmax(self.predictions, axis=1)
            self.species_mapping = {0:'Iris-Setosa', 1:'Iris-Versicolor', 2:'Iris-Virginica'}
            for i in self.class_labels:
                self.d.append(i)
            for i, label in enumerate(self.class_labels):
                print(f'Test Sample {i+1}: Predicted Species = {self.species_mapping[label]}')
            print(f'Classification report: \n {classification_report(self.y_test, self.d)}')
            print(f'Confusion matrix: \n {confusion_matrix(self.y_test, self.d)}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Message: {er_msg} \n Error Traceback: {er_line.tb_lineno}')

if __name__ == "__main__":
    try:
        ann = ANN_Multilabel()
        ann.architecture()
        ann.metrics()
        ann.visualization()
        ann.testing()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'Error Type: {er_type} \n Error Message: {er_msg} \n Error Traceback: {er_line.tb_lineno}')
