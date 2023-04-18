import pandas as pd
import numpy as np
from datetime import datetime
import os
import re

def identify_source_files(path="/Users/ryanschubert/Dropbox (Rush)/TREAT Lab/Wearables/Data/"):
    folders=os.listdir(path)
    folders=[f for f in folders if re.search('^([1-9])+(_)', f)]
    prefixes=[]
    for i  in enumerate(folders):
        files = os.listdir(path + i[1])
        files = [re.sub('\\.([a-zA-Z_-])+\\.([a-z])+$','',x) for x in files]
        files = set(files)
        files = [path + i for i in files]
        prefixes.append(files)
    prefixes = [file for sublist in prefixes for file in sublist]
    measurement_prefixes=[i for i in prefixes if re.search('([0-9]){8,8}$',i)]
    survey_prefixes=[i for i in prefixes if re.search('([0-9]){,8}(T)([0-9])+$',i)]
    return measurement_prefixes, survey_prefixes


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