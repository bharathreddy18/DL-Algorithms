import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.activations import relu, sigmoid
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix



class ANN_Binary:
    def __init__(self):
        try:
            self.df = pd.read_csv('breast_cancer.csv')
            self.df = self.df.drop(['id', 'Unnamed: 32'], axis=1)
            self.df['diagnosis'] = self.df['diagnosis'].map({'M':0, 'B':1}).astype(int)   #M-0, B-1
            self.X = self.df.iloc[:, 1:]
            self.y = self.df.iloc[:, 0]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Message: {er_msg} \n Error Traceback: {er_line.tb_lineno}')


    def architecture(self):
        try:
            self.model = Sequential()
            self.model.add(Dense(units = 128, kernel_initializer = 'he_uniform', activation = relu, input_dim = self.X_train.shape[1]))
            self.model.add(Dense(units = 64, kernel_initializer = 'he_uniform', activation = relu))
            self.model.add(Dense(units = 32, kernel_initializer = 'he_uniform', activation = relu))
            self.model.add(Dense(units = 16, kernel_initializer = 'he_uniform', activation = relu))
            self.model.add(Dense(units = 8, kernel_initializer = 'he_uniform', activation = relu))
            self.model.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = sigmoid))
            # print(self.model.summary())
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Message: {er_msg} \n Error Traceback: {er_line.tb_lineno}')


    def metrics(self):
        try:
            self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            self.model.fit(self.X_train, self.y_train, batch_size = 20, validation_split = 0.1, epochs = 50)
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
            self.predictions = self.model.predict(self.X_test)
            self.class_labels = (self.predictions>0.5).astype(int).flatten()
            for i, label in enumerate(self.class_labels):
                if label == 0:
                    print(f'Test Sample {i+1}: Malignant Cancer')
                else:
                    print(f'Test Sample {i+1}: Benign Cancer')
            print(f'Classification report: \n \n{classification_report(self.y_test, self.class_labels)}')
            print('----------------------------------------------------------------------------')
            print(f'Confusion matrix: \n \n{confusion_matrix(self.y_test, self.class_labels)}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type} \n Error Message: {er_msg} \n Error Traceback: {er_line.tb_lineno}')


if __name__ == '__main__':
    try:
        ann = ANN_Binary()
        ann.architecture()
        ann.metrics()
        ann.visualization()
        ann.testing()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'Error Type: {er_type} \n Error Message: {er_msg} \n Error Traceback: {er_line.tb_lineno}')


