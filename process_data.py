
import pandas as pd
import numpy as np
from datetime import datetime
import os

def identify_source_files(path="/Users/ryanschubert/Dropbox (Rush)/TREAT Lab/Wearables/Data/"):
    folders=os.listdir(path)
    for i  in enumerate(folders):
        print(i)
    return


def readInPPG(path):
    data=pd.read_csv(path)
    #convert timestamps
    data['ts']=datetime.utcfromtimestamp(data['ts']).strftime('%Y-%m-%d %H:%M:%S')
    return data

#get mood data


def identifyBounds():

    return
#remove sleep data



#window generation

#data augmentation

#take a random

def wearables_dataset():

    return [(train_data, train_labels),(test_data,test_labels)]