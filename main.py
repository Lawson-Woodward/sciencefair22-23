#===========================================================================================#
#IMPORTING VARIOUS LIBRARIES
#===========================================================================================#
import pandas as pd
import os
import pathlib
import tarfile
import gzip
import shutil
import logging
import re
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import sklearn

from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


#===========================================================================================#
#DEFINING IMPORTANT VARIABLES
#===========================================================================================#

full_data_dir = 'C:/eeg/sciencefair22-23/data/eeg_full/'
processing_dir = 'processed'
processed_data_dir = os.path.join(full_data_dir, processing_dir)

#we need to check these channel names still
channels = ['AF1', 'AF2','AF7','AF8','AFZ','FP1', 'FP2', 'CPZ', 'CZ', 'FCZ','FPZ', 'FT7', 'FT8', 'FZ','O1','O2','OZ','POZ','PZ','PO1','PO2','PO7','PO8', 'S1', 'T7','T8','TP7','TP8'] + [f'F{n:01}' for n in range(1,9)] + [f'C{n:01}' for n in range(1,8)] + [f'CP{n:01}' for n in range(1,7)] + [f'F{n:01}' for n in range(1,6)]  + [f'FC{n:01}' for n in range(1,7)] + [f'P{n:01}' for n in range(1,9)]

num_channels = 68
num_samples = 256

subject_dirs = pathlib.Path("C:\eeg\sciencefair22-23\data\eeg_full\processed").glob("co*")


#===========================================================================================#
#WRITING FUNCTIONS THAT WE WILL LATER CALL
#===========================================================================================#

def create_data_processing_dir(processed_data_dir):
    try:
        os.makedirs(processed_data_dir, exist_ok = True)
        print ("Target dir '%s' for processed data successfully created" % processed_data_dir)
    except OSError as error:
        print ("Directory '%s' cannot be created" % processed_data_dir)

def extract_subjects_to_dir(full_dir, processed_dir):
    for file in pathlib.Path(full_dir).glob('*.tar.gz'):
        input = tarfile.open(file)
        input.extractall(processed_dir)
        input.close()
        subject = tarfile.open(file, "r:gz")
        subject.extractall(processed_dir)
        subject.close()

def extract_files_to_subject(processed_dir, file):
    subject_dir = os.path.join(processed_dir, os.path.basename(file)).replace(".tar.gz", "")
    for subject_file in pathlib.Path(subject_dir).glob("*.gz"):
        try:
            with gzip.open(subject_file,'rb') as in_file:
                subject_out_file = subject_file.with_suffix('')
                with  open (subject_out_file, 'wb') as out_file:
                    shutil.copyfileobj(in_file, out_file)
            os.remove(subject_file)
        except:
            print("Couldn't print the file named", subject_file, "because of error.")
            continue

def create_array(file_name, channels):
    #create empty numpy array
    file_data = np.zeros([num_samples, num_channels], dtype=np.float64)
    #remove the unknown channels
    with open(file_name) as f:
        for line in f:
            newline = line.split(" ")
            if(newline[0] == "#" or newline[1] == "nd" or newline[1] == "Y" or
            newline[1] == "N" or newline[1] == "X"):
                pass
            else:
                index = channels.index(newline[1])
                file_data[int(newline[2]), int(index)] = newline[3]

    f.close()
    #print(file_data)

    dirname = os.path.dirname(os.path.abspath(file_name))
    fname = os.path.basename(file_name)
    new_fname = os.path.join(dirname, 'new_' + fname)
    my_df = pd.DataFrame(file_data)
    my_df.to_csv(new_fname, header=channels, index=False, lineterminator='\n')
    return new_fname

def read_trial_file(trial_file):
    print("reading file:", trial_file)
    trial_data_list = pd.read_csv(trial_file, sep='\s+', comment='#', header=0)
    print(trial_data_list.head())
    print(trial_data_list.shape)
    return(trial_data_list)

def clean_subject_data_files(subject_dirs):
    subjects = list()
    trial_files_data = list()
    for subject_dir in subject_dirs:
        print(subject_dir)
        #adds all files in the subject dir to the list
        subjects.append(os.path.basename(subject_dir))
        print("subject is", subjects[-1])
        trial_files = pathlib.Path(subject_dir).glob("co*")
        print("trial files are: ")
        for trial_file in trial_files:
            print(trial_file)
            clean_file = create_array(trial_file, channels)
            trial_files_data.append(read_trial_file(clean_file))
        print("trial fiels completed for subject")
        print ("trial file 0 was", trial_files_data[0])
        print ("Last trial file was", trial_files_data[-1])
        print("total number of subjects:", len(subjects))
    len_trial_files_data = []
    for trial in trial_files_data:
        len_trial_files_data.append(len(trial))
    #all samples are the exact same length, so we have it easy here
    print(pd.Series(len_trial_files_data).describe())

def create_targets_file(subject_dirs):
    targets=list()
    for subject_dir in subject_dirs:
        print(subject_dir)
        trial_files = pathlib.Path(subject_dir).glob("new*")
        a_or_c = os.path.basename(subject_dir)[3]
        if a_or_c == "a":
            a_or_c = "1"
        else:
            a_or_c = "0"
        print("subject is", a_or_c)
        for trial_file in trial_files:
            fn = os.path.basename(trial_file)
            x = [fn, a_or_c]
            print(x)
            targets.append(x)
        my_df = pd.DataFrame(targets)
        fn = os.path.join(processed_data_dir, 'targets.csv')
        print("added", fn)
        my_df.to_csv(fn, header=['#sequence_ID', 'class_label'], index=False, lineterminator='\n')

def create_groups_file(subject_dirs):
    targets = list()
    a = list()
    c = list()
    for subject_dir in subject_dirs:
        print(subject_dir)
        trial_files = pathlib.Path(subject_dir).glob("new*")
        a_or_c = os.path.basename(subject_dir)[3]
        if a_or_c == "a":
            a.append(subject_dir)
        else:
            c.append(subject_dir)

    subjects = a + c
    count=1
    for subject_dir in subjects:
        trial_files = pathlib.Path(subject_dir).glob("new*")
        for trial_file in trial_files:
            fn = os.path.basename(trial_file)
            x = [fn, count]
            print(x)
            targets.append(x)
        my_df = pd.DataFrame(np.array(targets))
        fn = os.path.join(processed_data_dir, 'groups.csv')
        print("added", fn)
        my_df.to_csv(fn, header=['#sequence_ID', 'dataset_ID'], index=False, lineterminator='\n')
        if count == 3:
            count = 0
        count += 1


#===========================================================================================#
#CALLING ALL OF THE FUNCTIONS THAT WERE WRITTEN SO FAR
#===========================================================================================#

  #created the processing directory:
#create_data_processing_dir(processed_data_dir)

  #extracted each subject to the processing direcotry
#extract_subjects_to_dir(full_data_dir, processed_data_dir)

  #extracted all of the subject files for each subject the processing directory:
#for file in pathlib.Path(processed_data_dir).glob("co*"):
#   print ("file is %s" % file)
#   extract_files_to_subject(processed_data_dir, file)

  #create the array for a datafile within the subject files, then put it in a text file (and read it)
#create_array("C:/eeg/sciencefair22-23/data/eeg_full/processed/co2a0000364/co2a0000364.rd.000", channels)
#read_trial_file("C:/eeg/sciencefair22-23/data/eeg_full/processed/co2a0000364/new_co2a0000364.rd.000")

  #creates a cleaned up version of every single trial file in the subject files provided
#clean_subject_data_files(subject_dirs)

  #creates the file that labels all subjects as a/c
#create_groups_file(subject_dirs)

  #creates the file that equally distributes the a/c files among 3 groups
#create_targets_file(subject_dirs)


#===========================================================================================#
#READ IN DATA AND SEPARATE IT BASED ON GROUP INTO TRAIN, VALIDATION, AND TEST SETS
#===========================================================================================#

#datasetgroup_file =  os.path.join(processed_data_dir,'DataSetGroup.csv')
groupsFile =  os.path.join(processed_data_dir,'minigroups.csv')
#targets_file = os.path.join(processed_data_dir,'targets.csv')
targetsFile = os.path.join(processed_data_dir,'minitargets.csv')
trials = pd.read_csv(groupsFile)

train_raw = trials.loc[trials['dataset_ID']==1]
validation_raw = trials.loc[trials['dataset_ID']==2]
test_raw = trials.loc[trials['dataset_ID']==3]
print(train_raw)
print(validation_raw)
print(test_raw)

train = list()
validation = list()
test = list()

for i in train_raw.index:
    fn = train_raw['#sequence_ID'][i]
    dir = fn.replace('new_','')
    dir = re.sub(".rd.*",'', dir)
    infile = os.path.join(full_data_dir, processing_dir + '/' + dir + '/' + fn)
    df = pd.read_csv(infile, header=0)
    values = df.values
    train.append(values)

for i in validation_raw.index:
    fn = validation_raw['#sequence_ID'][i]
    dir = fn.replace('new_','')
    dir = re.sub(".rd.*",'', dir)
    infile = os.path.join(full_data_dir, processing_dir + '/' + dir + '/' + fn)
    df = pd.read_csv(infile, header=0)
    values = df.values
    validation.append(values)

for i in test_raw.index:
    fn = test_raw['#sequence_ID'][i]
    dir = fn.replace('new_','')
    dir = re.sub(".rd.*",'', dir)
    infile = os.path.join(full_data_dir, processing_dir + '/' + dir + '/' + fn)
    df = pd.read_csv(infile, header=0)
    values = df.values
    test.append(values)

train=np.array(train).round(5)
validation=np.array(validation).round(5)
test=np.array(test).round(5)
np.set_printoptions(suppress=True)
print('\n\nTRAIN SET:\n', train)
print('\n\nVALIDATION SET:\n', validation)
print('\n\nTEST SET:\n', test)


#===========================================================================================#
#CREATE TARGET VALUE ARRAYS TO MATCH THE TRAIN, VALIDATION, AND TEST ARRAYS
#===========================================================================================#

targetsInfo = pd.read_csv(targetsFile)
groupsInfo = pd.read_csv(groupsFile)
mergedInfo = pd.merge(targetsInfo, groupsInfo, on='#sequence_ID', how='outer')
print(mergedInfo)

raw_train_target = mergedInfo.loc[mergedInfo['dataset_ID']==1]
raw_validation_target = mergedInfo.loc[mergedInfo['dataset_ID']==2]
raw_test_target = mergedInfo.loc[mergedInfo['dataset_ID']==3]

train_target = raw_train_target['class_label']
validation_target = raw_validation_target['class_label']
test_target = raw_test_target['class_label']

print(raw_train_target)
print(raw_validation_target)
print(raw_test_target)

train_target = np.array(train_target)
validation_target = np.array(validation_target)
test_target = np.array(test_target)

print(train_target)
print(validation_target)
print(test_target)


