#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:51:29 2023

@author: ryanschubert
"""

#1d conv neural netwodk

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

##import data

#train_data
#train_labels

#test_data
#test_labels
### define callback

class myCallback(tf.keras.callback.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('loss')<0.1):
            print("\nTarget loss achieved!")
            self.model.stop_training = True


##define neural net
#lets start with a simple version
#need to define the input shapes, which will depend on the actual data
#The configuration of the 1D CNN used in all experiments has 3 hidden convolutional layers and 2 dense layers. The 1D CNNs have 32 and 16 neurons on the first and second hidden convolutional layers and 10 neurons on the hidden dense layer. The output layer size is 5 which is the number of beat classes and the input (CNN) layer size is either 2 (base) or 4 (extended) according to the choice of raw data representation. For 64 and 128 sample beat representations, the kernel sizes are set to 9 and 15, and the sub-sampling factors are set to 4 and 6, respectively
def define_model():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(),#need to decide how many filters and define the input shape
            tf.keras.layers.MaxPool1D(), #need to decide 
            tf.keras.layers.Dense(), #do i need a flatten layer?
        ])

    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return(model)



callbacks = myCallback()

model = define_model()

# train model
history = model.fit(train_data, train_labels, epochs=10, callbacks=[callbacks])