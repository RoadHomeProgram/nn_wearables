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
from random import choices
import pickle

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
Y = np.asarray(Y).astype(np.float32)
indices=[j for j in range(len(ids)) if ids[j] != 'Bcw9ieaoXiFLhRZGX2O7Xw']
X=np.delete(X,indices,axis=0)
Y=np.delete(Y,indices)


val_r2_hist=[]
val_loss_hist=[]
for i in range(1,len(Y)):
    X_holdin=np.delete(X,i,axis=0)
    X_holdin=scaler.fit_transform(X_holdin.reshape(-1, X_holdin.shape[-1])).reshape(X_holdin.shape)
    X_holdin=np.nan_to_num(X_holdin)
    Y_holdin=np.delete(Y,i)

    X_holdout=X[i,]
    X_holdout=X_holdout[np.newaxis,:,:]
    X_holdout=scaler.fit_transform(X_holdout.reshape(-1, X_holdout.shape[-1])).reshape(X_holdout.shape)
    X_holdout=np.nan_to_num(X_holdout)
    Y_holdout=np.asarray(Y[i,]).reshape((1,))
    model = define_model()
    history = model.fit(X_holdin,Y_holdin,validation_data=(X_holdout,Y_holdout),epochs=50,batch_size=5)
    
    plt.plot(history.history['r_square'])
    plt.plot(history.history['val_r_square'])
    plt.title('model r_square ' + str(i))
    plt.ylabel('r_square')
    plt.xlabel('epoch')
    plt.legend(['train N=' +str(Y_holdin.shape[0]), 'test N=' +str(Y_holdout.shape[0])], loc='upper left')
    plt.show()
    
    val_r2_hist.append(history.history['val_r_square'][-1])
    val_loss_hist.append(history.history['val_loss'][-1])
    

with open("/Users/ryanschubert/Dropbox (Rush)/Ryan's stuff/wearables/spotcheck_W8mmLTzDR0sK6ZocflG4KA.pkl", 'wb') as outp:
    pickle.dump((val_loss_hist,Y[1:36]), outp, pickle.HIGHEST_PROTOCOL)


a, b = np.polyfit(Y[1:36],val_loss_hist, 1)
plt.scatter(Y[1:36],val_loss_hist)
plt.plot(Y[1:36], a*Y[1:36]+b)
plt.show()
    