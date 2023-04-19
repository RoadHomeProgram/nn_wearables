import pandas as pd
import numpy as np
from datetime import datetime
import os
import re

def identify_patients(file="/Users/ryanschubert/Dropbox (Rush)/Ryan's stuff/wearables/patient_mapping.csv"):
    data = pd.read_csv(file)
    data = data.loc[data['V4'] == 'T']
    return(data['V3'].tolist())

def identify_source_files(path="/Users/ryanschubert/Dropbox (Rush)/TREAT Lab/Wearables/Data/"):
    folders=os.listdir(path)
    folders=[f for f in folders if re.search('^([1-9])+(_)', f)]
    valid_patients = identify_patients()
    prefixes=[]
    for i  in enumerate(folders):
        files = os.listdir(path + i[1])
        files = [f for f in files if re.search('ppg.csv',f) or re.search('json',f)]
        files = [re.sub('\\.([a-zA-Z_-])+\\.([a-z])+$','',x) for x in files]
        files = [f for f in files if re.sub('([a-zA-Z_0-9]+)(\\.)([0-9T])+','\\1',f) in valid_patients]
        files = set(files)
        files = [path + i[1] + '/' + f for f in files]
        prefixes.append(files)
    prefixes = [file for sublist in prefixes for file in sublist]
    measurement_prefixes=[i for i in prefixes if re.search('([0-9]){8,8}$',i)]
    survey_prefixes=[i for i in prefixes if re.search('([0-9]){,8}(T)([0-9])+$',i)]
    return measurement_prefixes, survey_prefixes


def readPPG(path):
    data=pd.read_csv(path)
    # #convert timestamps
    data['ts']=data['ts']/1000000000
    data['ts']=[datetime.utcfromtimestamp(date) for date in data['ts']]
    return data

def mapMoodToPPG(prefix,moodSurveys):
    cleanRegex=re.sub('\\)','\\)',re.sub('\\(','\\(',re.sub(' ' ,'\\ ',prefix)))
    targetSurveys=[f for f in moodSurveys if re.search('(' + cleanRegex +')',f)]
    return targetSurveys

#get mood data
def readMood():
    return

def identifyBounds():

    return
#remove sleep data



#window generation

#data augmentation

#take a random

def wearables_dataset():

    return [(train_data, train_labels),(test_data,test_labels)]