#===========================================================================================#
#IMPORTING VARIOUS LIBRARIES
#===========================================================================================#
import pandas as pd
import os
import pathlib  #Deals with paths like OS
import tarfile  #Allows to open the tarFile
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
processed_data_dir = os.path.join(full_data_dir, processing_dir) #Creating a new file path called, C:/eeg/sciencefair22-23/data/eeg_full/processed

#Important for Machine Learning, specifes paramaters
#channels = ['AF1', 'AF2','AF7','AF8','AFZ','FP1', 'FP2', 'CPZ', 'CZ', 'FCZ','FPZ', 'FT7', 'FT8', 'FZ','O1','O2','OZ','POZ','PZ','PO1','PO2','PO7','PO8', 'S1', 'T7','T8','TP7','TP8'] + [f'F{n:01}' for n in range(1,9)] + [f'C{n:01}' for n in range(1,8)] + [f'CP{n:01}' for n in range(1,7)] + [f'F{n:01}' for n in range(1,6)]  + [f'FC{n:01}' for n in range(1,7)] + [f'P{n:01}' for n in range(1,9)]
channels = ['FP1','FP2','F7','F8','AF1','AF2','FZ','F4','F3','FC6','FC5','FC2','FC1','T8','T7','CZ','C3','C4','CP5','CP6','CP1','CP2','P3','P4','PZ','P8','P7','PO2','PO1','O2','O1','AF7','AF8','F5','F6','FT7','FT8','FPZ','FC4','FC3','C6','C5','F2','F1','TP8','TP7','AFZ','CP3','CP4','P5','P6','C1','C2','PO7','PO8','FCZ','POZ','OZ','P2','P1','CPZ',]   #excluding 'nd','Y','X'

#num_channels = 61
num_channels = 61
num_samples = 256

subject_dirs = pathlib.Path("C:\eeg\sciencefair22-23\data\eeg_full\processed").glob("co*") #Finding all files that starts with co


#===========================================================================================#
#WRITING FUNCTIONS THAT WE WILL LATER CALL
#===========================================================================================#

def create_data_processing_dir(processed_data_dir): #If no folder named processing, it creates one to store in the processed data later
    try:
        os.makedirs(processed_data_dir, exist_ok = True)
        print ("Target dir '%s' for processed data successfully created" % processed_data_dir)
    except OSError as error:
        print ("Directory '%s' cannot be created" % processed_data_dir)

def extract_subjects_to_dir(full_dir, processed_dir): #For every single file that ends with tar.gz it extracts into the processed directory
    for file in pathlib.Path(full_dir).glob('*.tar.gz'):
        input = tarfile.open(file) 
        input.extractall(processed_dir) #Untars and unzips it
        input.close()

        subject = tarfile.open(file, "r:gz") #It extracts to the processed directory
        subject.extractall(processed_dir)
        subject.close()

#processed dir > patients (co) > # of trials > list of electrodes
def extract_files_to_subject(processed_dir, file): #Uncompresses which removes the gz file extension, is called in a for loop
    subject_dir = os.path.join(processed_dir, os.path.basename(file)).replace(".tar.gz", "")
    for subject_file in pathlib.Path(subject_dir).glob("*.gz"): #subject_file = trials
        try:
            with gzip.open(subject_file,'rb') as in_file:
                subject_out_file = subject_file.with_suffix('')
                with  open (subject_out_file, 'wb') as out_file:
                    shutil.copyfileobj(in_file, out_file)
            os.remove(subject_file)
        except:
            print("Couldn't print the file named", subject_file, "because of error.")
            continue

def create_array(file_name, channels): #Called in a for loop, creates an array for all the data
    #creates empty numpy array
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

    #Puts all the data into a csv file
    dirname = os.path.dirname(os.path.abspath(file_name)) 
    fname = os.path.basename(file_name)
    new_fname = os.path.join(dirname, 'new_' + fname)
    my_df = pd.DataFrame(file_data)
    my_df.to_csv(new_fname, header=channels, index=False, lineterminator='\n')
    return new_fname

def read_trial_file(trial_file): #Getting the data
    print("reading file:", trial_file) #trial_file is a csv file
    trial_data_list = pd.read_csv(trial_file, sep='\s+', comment='#', header=0)
    print(trial_data_list.head())
    print(trial_data_list.shape)
    return(trial_data_list)

def clean_subject_data_files(subject_dirs): #Uses the above methods to actually move and change the data. Creates the new_co files
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

def create_targets_file(subject_dirs): #For every new file created, you are reading the 4th character of the file and making it binary. Control = 0, Alcohol = 1
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

def create_groups_file(subject_dirs): #Creates an A and C list, goes through each subject folder, and determined alchohol or control
    targets = list()
    a = list()
    c = list()
    for subject_dir in subject_dirs: #Sorts into a then c   
        print(subject_dir)
        trial_files = pathlib.Path(subject_dir).glob("new*")
        a_or_c = os.path.basename(subject_dir)[3]
        if a_or_c == "a":
            a.append(subject_dir)
        else:
            c.append(subject_dir)
    #1 = train, 2 = validation, 3 = test
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

# train=np.array(train).round(5)
# validation=np.array(validation).round(5)
# test=np.array(test).round(5)
# np.set_printoptions(suppress=True)
# print('\n\nTRAIN SET:\n', train)
# print('\n\nVALIDATION SET:\n', validation)
# print('\n\nTEST SET:\n', test)


#===========================================================================================#
#CREATE TARGET VALUE ARRAYS TO MATCH THE TRAIN, VALIDATION, AND TEST ARRAYS
#===========================================================================================#

targetsInfo = pd.read_csv(targetsFile)
groupsInfo = pd.read_csv(groupsFile)
mergedInfo = pd.merge(targetsInfo, groupsInfo, on='#sequence_ID', how='outer')
print(mergedInfo)

# within each group is a binary indicator as to whether or not the patient was alcohol(1)/control(0)
raw_train_target = mergedInfo.loc[mergedInfo['dataset_ID']==1] # group 1 = training set
raw_validation_target = mergedInfo.loc[mergedInfo['dataset_ID']==2] # group  2 = validtion set
raw_test_target = mergedInfo.loc[mergedInfo['dataset_ID']==3] # group 3 = test set 

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


#===========================================================================================#
#BUILD A TIME SERIES CLASSIFICATION MODEL AND A SINGLE LAYER LSTM MODEL
#===========================================================================================#

best_model_pkl = os.path.join(processed_data_dir, 'best_model.pkl')

model = Sequential()
model.add(LSTM(256, input_shape=(num_samples, num_channels)))
model.add(Dense(1, activation='sigmoid'))
print("*** Model Summary *** ")
print(model.summary())

adam = Adam(learning_rate=0.001)
chk = ModelCheckpoint(best_model_pkl, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

history = model.fit(train, train_target, epochs=10, batch_size=128, callbacks=[chk], validation_data=(validation,validation_target))
#model.fit(train, train_target, epochs=200, batch_size=128, callbacks=[chk], validation_data=(validation,validation_target))
### 12/27 model.fit(train, train_target, epochs=1, batch_size=1, callbacks=[chk], validation_data=(validation,validation_target))
 
# summarize data for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
 
#summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model = load_model(best_model_pkl)

from sklearn.metrics import accuracy_score
test_preds = (model.predict(test) > 0.5).astype("int32")
print("Accuracy Score: ", accuracy_score(test_target, test_preds))
 
exit()

