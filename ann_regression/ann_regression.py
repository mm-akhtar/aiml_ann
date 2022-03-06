# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:05:24 2022

@author: kkakh
"""
# Artificial neural network 

# importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# part-1 Data preprocessing
    # importing the dataset
dataset = pd.read_excel("Folds5x2_pp.xlsx")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
    # spliting the dataset into the training dataset and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0)

# part-2 Building the ANN
    # initializing the ANN
    ann = tf.keras.models.Sequential()
    # adding the input layer and 1st hidden
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    # adding the 2nd hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    # adding the output layer
    ann.add(tf.keras.layers.Dense(units=1))

# part-3 Training the ANN
    # compiling the ann (for non-binary output loss= categorical_crossentropy)
    ann.compile(optimizer='adam' , loss='mean_squared_error')
    # training the ann on training set
    ann.fit(X_train, y_train, batch_size= 32, epochs=100)
    
# part-4 making the prediction and evaluating the model
    # predicting the test set results
    y_pred = ann.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

