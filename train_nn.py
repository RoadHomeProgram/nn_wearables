#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:51:29 2023

@author: ryanschubert
"""

#1d conv neural netwodk
import os
os.chdir('/Users/ryanschubert/Documents/wearables/nn_wearables/')
import tensorflow as tf
import process_data as ppd
import numpy as np
# import keras
##import data
X, Y = ppd.wearables_dataset()
X = np.asarray(X).astype(np.float32)
X = nan_to_num(X)
# class myCallback(keras.callback.Callback):
#     def on_epoch_end(self,epoch,logs={}):
#         if(logs.get('loss')<0.1):
#             print("\nTarget loss achieved!")
#             self.model.stop_training = True

n_steps=1000
n_features=2
##define neural net
#lets start with a simple version
#need to define the input shapes, which will depend on the actual data
#The configuration of the 1D CNN used in all experiments has 3 hidden convolutional layers and 2 dense layers. The 1D CNNs have 32 and 16 neurons on the first and second hidden convolutional layers and 10 neurons on the hidden dense layer. The output layer size is 5 which is the number of beat classes and the input (CNN) layer size is either 2 (base) or 4 (extended) according to the choice of raw data representation. For 64 and 128 sample beat representations, the kernel sizes are set to 9 and 15, and the sub-sampling factors are set to 4 and 6, respectively
def define_model():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(32, kernel_size=9, activation='relu', input_shape=(n_steps, n_features)),#need to decide how many filters and define the input shape
            tf.keras.layers.Conv1D(16, kernel_size=9, activation='relu'),#need to decide how many filters and define the input shape
            tf.keras.layers.MaxPool1D(pool_size=4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50,activation='relu'),
            tf.keras.layers.Dense(11,activation='softmax'),
        ])

    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return(model)

# callbacks = myCallback()

model = define_model()

# train model
history = model.fit(X, Y, epochs=10,batch_size=10)

