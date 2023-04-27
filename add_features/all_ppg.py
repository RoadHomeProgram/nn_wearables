import os 
os.chdir('/Users/ryanschubert/Documents/wearables/nn_wearables/')
import tensorflow as tf
import process_data as ppd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_addons.metrics import RSquare
from sklearn.preprocessing import StandardScaler


n_steps=5000
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


uniqueIds=set(ids)

for i in uniqueIds:
    print(i)
    indices=[j for j in range(len(ids)) if ids[j] != i]
    indiv_X=np.delete(X,indices,axis=0)
    indiv_X= scaler.fit_transform(indiv_X.reshape(-1, indiv_X.shape[-1])).reshape(indiv_X.shape)
    indiv_X=np.nan_to_num(indiv_X)
    indiv_Y=np.delete(Y,indices)
    model = define_model()
    history = model.fit(indiv_X, indiv_Y, validation_split=0.3,epochs=200,batch_size=10)
    
    plt.plot(history.history['r_square'])
    plt.plot(history.history['val_r_square'])
    plt.title('model r_square ' + i + ' N=' + str(indiv_Y.shape[0]))
    plt.ylabel('r_square')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
   
