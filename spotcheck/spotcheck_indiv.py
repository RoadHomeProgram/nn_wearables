#W8mmLTzDR0sK6ZocflG4KA has a particularly good fit in this model
#want to make sure this is not an artifact so will do k fold crossvalidation on their data alone to estimate the fit of the neural net
import os 
os.chdir('/Users/ryanschubert/Documents/wearables/nn_wearables/')
import tensorflow as tf
import process_data as ppd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_addons.metrics import RSquare
from sklearn.preprocessing import StandardScaler
from random import sample

n_steps=10000
n_features=4
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
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(50,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(50,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1,activation='relu'),
        ])

    model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=[RSquare()])
    return(model)

scaler = StandardScaler()

X, Y, ids = ppd.wearables_dataset(n=n_steps)
X = np.asarray(X).astype(np.float32)
indices=[j for j in range(len(ids)) if ids[j] != 'W8mmLTzDR0sK6ZocflG4KA']
X=np.delete(X,indices,axis=0)
Y=np.delete(Y,indices)

fold_ids=sample((1,2,3,4),Y.shape[0])


model = define_model()
history = model.fit(indiv_X, indiv_Y,epochs=200,batch_size=5,validation_split=0.3)