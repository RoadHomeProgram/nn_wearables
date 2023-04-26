#here the question is how long of a window should we select
#I started with 1000 ts, but how much time does that potentially cover?
import os 
os.chdir('/Users/ryanschubert/Documents/wearables/nn_wearables/')
import tensorflow as tf
import process_data as ppd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

n_steps=10000
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
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(50,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(50,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(11,activation='softmax'),
        ])

    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return(model)



scaler = StandardScaler()

# train model


#hmm we need some code to handle the train test split
X, Y, ids = ppd.wearables_dataset(n=n_steps)
X = np.asarray(X).astype(np.float32)
X = np.nan_to_num(X)

uniqueIds=set(ids)

for i in uniqueIds:
    print(i)
    indices=[j for j in range(len(ids)) if ids[j] == i]
    train_X=np.delete(X,indices,axis=0)
    train_X= scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    train_Y=np.delete(Y,indices)
    test_X=X[indices,]
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    test_Y=Y[indices,]
    model = define_model()
    history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y),epochs=50,batch_size=10)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy ' + i)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
tmp=model.predict(test_X)
test_Y
