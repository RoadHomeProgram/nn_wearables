import os 
os.chdir('/Users/ryanschubert/Documents/wearables/nn_wearables/')
import tensorflow as tf
import process_data as ppd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_addons.metrics import RSquare
from sklearn.preprocessing import StandardScaler

#given that this neural net seems to have poor transferability between persons, lets check how this nn does within subjects
#essentially can we predict a person's future anxiety states given their previous anxiety states
n_steps=1000
n_features=2

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
            tf.keras.layers.Dense(1,activation='relu'),
        ])
    model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=[RSquare()])
    return(model)

scaler = StandardScaler()

#hmm we need some code to handle the train test split
X, Y, ids = ppd.wearables_dataset()
X = np.asarray(X).astype(np.float32)
X = np.nan_to_num(X)

uniqueIds=set(ids)

for i in uniqueIds:
    print(i)
    indices=[j for j in range(len(ids)) if ids[j] == i]
    indiv_X=np.delete(X,indices,axis=0)
    indiv_X= scaler.fit_transform(indiv_X.reshape(-1, indiv_X.shape[-1])).reshape(indiv_X.shape)
    indiv_Y=np.delete(Y,indices)
    model = define_model()
    history = model.fit(indiv_X, indiv_Y, validation_split=0.3,epochs=50,batch_size=10)
    
    plt.plot(history.history['r_square'])
    plt.plot(history.history['val_r_square'])
    plt.title('model r_square')
    plt.ylabel('r_square')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
   
(sum((np.argmax(model.predict(X),axis=1) - Y)**2)/124)
