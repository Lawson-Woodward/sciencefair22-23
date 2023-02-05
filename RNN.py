#===========================================================================================#
#IMPORTING VARIOUS LIBRARIES
#===========================================================================================#

import pandas as pd
import os
import pathlib  #Deals with paths like OS
import tarfile  #Allows to open the tarFile
import gzip
import shutil
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datetime import datetime
from keras.utils.vis_utils import plot_model

from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import sys

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

np.set_printoptions(threshold=sys.maxsize)

#===========================================================================================#
#DEFINING IMPORTANT VARIABLES
#===========================================================================================#

do_initial_extraction = 0 #Only need to use this once to extract the original data
process_raw_data = 0      #Only need to use this when you change the selected data (s1 obj, s2 match, s2 nomatch)
 
full_data_dir = 'C:/eeg/sciencefair22-23/data/eeg_full/'  
 
processing_dir = 'processed' #manually clean up the new_ files in this dir by searching for name:new_* and deleting
processed_data_dir = os.path.join(full_data_dir, processing_dir) #Creating a new file path called, C:/eeg/sciencefair22-23/data/eeg_full/processed
groupsFile =  os.path.join(processed_data_dir,'groups.csv')
trial_files_descriptions = os.path.join(processed_data_dir, 'trial_files_descriptions.csv')
targetsFile = os.path.join(processed_data_dir,'targets.csv')
experimental_dir = os.path.join(full_data_dir, 'experiments/')

#Important for Machine Learning, specifes paramaters
channels = ['AF1', 'AF2','AF7','AF8','AFZ','FP1', 'FP2', 'CPZ', 'CZ', 'FCZ','FPZ', 'FT7', 'FT8', 'FZ','O1','O2','OZ','POZ','PZ','PO1','PO2','PO7','PO8', 'T7','T8','TP7','TP8'] + [f'F{n:01}' for n in range(1,9)] + [f'C{n:01}' for n in range(1,7)] + [f'CP{n:01}' for n in range(1,7)] + [f'FC{n:01}' for n in range(1,7)] + [f'P{n:01}' for n in range(1,9)]

num_channels = len(channels)
num_samples = 256

subject_dirs=list(pathlib.Path(processed_data_dir).glob("co*")) #Finding all files that starts with co

# Experimental Values
datasubset='S' #'S' is for all, 'S1 obj', 'S2 match' or 'S2 nomatch' are specific stimuli

lr=[.0005]     # you can turn any of the following 4 values into a list of values, and then
lstm_units=256 # change the for statement on line 292 to run though all of the values for the
epochs= 20     # specific variable you want to change. It runs through all values of the list
batch_size=256 # in one run and outputs them to an 'experiments' folder
 
n_train = .6   #percent of data to use for training and validation
n_validate = .2
 
#===========================================================================================#
#WRITING FUNCTIONS THAT WE WILL LATER CALL
#===========================================================================================#
 
def create_data_processing_dir(processed_data_dir): #If no folder named processing, creates one to store the later processed data
    try:
        os.makedirs(processed_data_dir, exist_ok = True)
        print ("Target dir '%s' for processed data successfully created" % processed_data_dir)
    except OSError as error:
        print ("Directory '%s' cannot be created" % processed_data_dir)
 
def create_experimental_dir(experimental_dir): #If no folder named experiments, it creates one to store in the processed data later
    try:
        os.makedirs(experimental_dir, exist_ok = True)
        print ("Target dir '%s' for processed data successfully created" % experimental_dir)
    except OSError as error:
        print ("Directory '%s' cannot be created" % experimental_dir)
 
def extract_subjects_to_dir(full_data_dir, processed_data_dir): #For every single file that ends with tar.gz it extracts into the processed directory
    print ("extracting subject files from tar.gz")
    for file in pathlib.Path(full_data_dir).glob('*.tar.gz'):
        input = tarfile.open(file)
        input.extractall(processed_data_dir) #Untars and unzips it
        input.close()
        subject = tarfile.open(file, "r:gz") #It extracts to the processed directory
        subject.extractall(processed_data_dir)
        subject.close()
 
#processed dir > patients (co) > # of trials > list of electrodes <----- for size reference
def extract_files_to_subject(processed_data_dir, file): #Uncompresses which removes the gz file extension, is called in a for loop
    print ("extracting files to subject")
    subject_dir = os.path.join(processed_data_dir, os.path.basename(file)).replace(".tar.gz", "")
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
 
def create_clean_data_file(filename, ch_names, create_file,datasubset):
 
    # This let's us decide if we want to create clean data files for all the original data files
    # or if we want to focus on a subset, S1 bj, S2 match, S2 nomatch
    # If the original data file doesn't match, we skip it - no file gets created.
    create_file=0
    # Start by creating an empty 2d array that we can fill with data from a file
    data_array = np.zeros([num_samples, num_channels], dtype=np.float64)
 
    with open(filename) as f:
        content = f.readlines()
        if (datasubset in content[3]):
            create_file=1
            for line in content:
                newline = line.split()
                # Remove unknown channels. We were unable to determine what nd, X, and
                # Y stood for since they are nonstandard channels.
                # We also need to exclude comments.
                if (newline[1] in ch_names and newline[0] != "#"):
                    index = ch_names.index(newline[1])
                    # load the data into the np array
                    data_array[int(newline[2]), int(index)] = newline[3]
        else:
            create_file=0
    f.close()
 
    # we prepend new_ so that we can easily identify and work with the cleaned data files. We
    # don't replace or remove the originals, so that we have them for reference.
    if (create_file):
        dirname = os.path.dirname(os.path.abspath(filename))
        fname = os.path.basename(filename)
        newname = os.path.join(dirname,'new_' + fname)
        my_df = pd.DataFrame(data_array)
        my_df.to_csv(newname,header=ch_names, index=False, lineterminator='\n')  
        # This is our loaded data, transformed into an np array of channels rows x samples columns
        return newname
 
def read_trial_file(trial_file): # Getting the data
    trial_data_list = pd.read_csv(trial_file, sep='\s+', comment='#', header=0)
    return(trial_data_list)
 
def clean_subject_data_files(subject_dirs,datasubset): #Uses the above methods to actually move and change the data. Creates the new_co files
    # This identifies all of the trial files in a directory and sends them off to be cleaned
    print ("cleaning subject data files")
    subjects = list()
 
    trial_files_data = list()
    for subject_dir in subject_dirs:
        #adds all files in the subject dir to the list
        subjects.append(os.path.basename(subject_dir))
        trial_files = pathlib.Path(subject_dir).glob("co*")
        for trial_file in trial_files:
            clean_file = create_clean_data_file(trial_file, channels, 1,datasubset)
            if clean_file:
                trial_files_data.append(read_trial_file(clean_file))
 
    len_trial_files_data = []
    for trial in trial_files_data:
        len_trial_files_data.append(len(trial))
 
def map_files_to_stype_and_aorc(subject_dirs):
    print ("mapping files to s type and class")
    tfd_if = open(trial_files_descriptions, "a")  # append mode
 
    trial_files=[]
    for subject_dir in subject_dirs:
        fq_trial_files = pathlib.Path(subject_dir).glob("new*")
        for trial_file in fq_trial_files:
            base_fn = os.path.basename(trial_file)
            a_or_c = base_fn[7]
            if a_or_c == "a":
                a_or_c = "1"
            else:
                a_or_c = "0"
            orig_file = str(trial_file).replace('new_','')
            with open(orig_file) as f:
                content = f.readlines()
                if ('S1 obj' in content[3]):
                    s_type='S1 obj'
                elif ('S2 match' in content[3]):
                    s_type = 'S2 match'
                elif ('S2 nomatch' in content[3]):
                    s_type = 'S2 nomatch'
                else:
                    s_type = 'S type not found'
                datastring = str(trial_file) + ', '+ str(base_fn) + ', '+ str(a_or_c) +', '+ str(s_type)+'\n'
                tfd_if.write(datastring)
    tfd_if.close()
   
def create_groups_and_targets_file(trial_files_descriptions, datasubset):
    tfd = pd.read_csv(trial_files_descriptions,sep=",")
    tfd.columns=['FQN File', '#sequence_ID', 'class_label', 's_type']
    #filter for only the desired s_type
    tfd = tfd[tfd['s_type'].str.contains(datasubset)]
    print("stype rows are", tfd.shape[0])
   
    # this code gets two pds with equal number of 0 and 1 by removing extras.
    # It randomizes the lists before removing extras, so the same subset of data isn't chosen repeatedly.
    c_tfd = tfd.loc[tfd['class_label'] == 0]
    a_tfd = tfd.loc[tfd['class_label'] == 1]
    a_tfd = a_tfd.sample(frac = 1)
    c_tfd = c_tfd.sample(frac = 1)
    c_rows = c_tfd.shape[0]
    a_rows = a_tfd.shape[0]
    diff = c_rows - a_rows
    if diff < 0:
        a_tfd = a_tfd.iloc[:c_rows]
    else:
        c_tfd = c_tfd.iloc[:a_rows]
   
    #### You need to split these into 60/20/20 PDs and then combine the PDs.
    train_rows=int(n_train*c_tfd.shape[0])
    a_tfd_train=a_tfd.iloc[:train_rows]
    val_rows=int(n_validate*c_tfd.shape[0])
    endval=int(train_rows+val_rows)
    a_tfd_val=a_tfd.iloc[train_rows:endval]
    a_tfd_test=a_tfd.iloc[endval:]
 
    c_tfd_train=c_tfd.iloc[:train_rows]
    c_tfd_val=c_tfd.iloc[train_rows:endval]
    c_tfd_test=c_tfd.iloc[endval:]
 
    data=[a_tfd_train,c_tfd_train]
    tfd_train=pd.concat(data)
    data=[a_tfd_val,c_tfd_val]
    tfd_val=pd.concat(data)
    data=[a_tfd_test,c_tfd_test]
    tfd_test=pd.concat(data)
 
    #randomize again, so we don't have all 1's and all 0's clumped together after the concat
    tfd_train = tfd_train.sample(frac = 1)
    tfd_train['dataset_ID'] = '1'
    tfd_val = tfd_val.sample(frac = 1)
    tfd_val['dataset_ID'] = '2'
    tfd_test = tfd_test.sample(frac = 1)
    tfd_test['dataset_ID'] = '3'
    #create the files
    print(tfd_test)
 
    groups_fn = os.path.join(processed_data_dir, 'groups.csv')
    header=['#sequence_ID', 'dataset_ID']
    tfd_train.to_csv(groups_fn, columns = header, index=False, lineterminator='\n')
    tfd_val.to_csv(groups_fn, columns = header, mode='a', header=False,index=False)
    tfd_test.to_csv(groups_fn, columns = header, mode='a', header=False,index=False)
 
    targets_fn = os.path.join(processed_data_dir, 'targets.csv')
    header=['#sequence_ID', 'class_label']
    tfd_train.to_csv(targets_fn, columns = header, index=False, lineterminator='\n')
    tfd_val.to_csv(targets_fn, columns = header, mode='a', header=False,index=False)
    tfd_test.to_csv(targets_fn, columns = header, mode='a', header=False,index=False)
 
 
#===========================================================================================#
#CALLING ALL OF THE FUNCTIONS THAT WERE WRITTEN SO FAR
#===========================================================================================#
 
if do_initial_extraction == 1:
    #create the processing directory:
    create_data_processing_dir(processed_data_dir)
 
    #create directory to store experiment information
    create_experimental_dir(experimental_dir)
   
    #extracted each subject to the processing directory
    extract_subjects_to_dir(full_data_dir, processed_data_dir)
 
    for file in pathlib.Path(processed_data_dir).glob("co*"):
       print ("file is %s" % file)
       extract_files_to_subject(processed_data_dir, file)
 
if process_raw_data == 1:
    #creates a cleaned up version of every single trial file in the subject files provided
    #sets up the mapping of files to stype and a or c
    clean_subject_data_files(subject_dirs,datasubset)
    map_files_to_stype_and_aorc(subject_dirs)
 
# UNCOMMENT LINE 287 for the FIRST TIME that you create the new subject files...
# also COMMENT THE LINE OUT after the first run:
#create_groups_and_targets_file(trial_files_descriptions, datasubset)
 
for lr in (lr):
    print ("lr is", lr)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    print("timestamp is", timestamp)
    best_model_pkl = os.path.join(experimental_dir, timestamp+'_'+'best_model.pkl')
 
    #===========================================================================================#
    #READ IN DATA AND SEPARATE IT BASED ON GROUP INTO TRAIN, VALIDATION, AND TEST SETS
    #===========================================================================================#
 
    trials = pd.read_csv(groupsFile)
 
    #dataframes for each group
    train_raw = trials.loc[trials['dataset_ID']==1]
    validation_raw = trials.loc[trials['dataset_ID']==2]
    test_raw = trials.loc[trials['dataset_ID']==3]
   
    print("train_raw df is this long",len(train_raw.index))
    print("validation_raw df is this long",len(validation_raw.index))
    print("test_raw df is this long",len(test_raw.index))
 
    train = list()
    validation = list()
    test = list()
 
    count=len(train_raw.index)
    for i in range(count):
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
 
    print ("number of train rows is", train.shape[0])
    print ("number of validation rows is", validation.shape[0])
    print ("number of test rows is", test.shape[0])
 
    #===========================================================================================#
    #CREATE TARGET VALUE ARRAYS TO MATCH THE TRAIN, VALIDATION, AND TEST ARRAYS
    #===========================================================================================#
 
    targetsInfo = pd.read_csv(targetsFile)
    groupsInfo = pd.read_csv(groupsFile)
    mergedInfo = pd.merge(targetsInfo, groupsInfo, on='#sequence_ID', how='outer')
    print("merged info size is", len(mergedInfo.index))
    mi_df = pd.DataFrame(mergedInfo)
    mi_fn = os.path.join(processed_data_dir, 'mergedinfo.csv')
    mi_df.to_csv(mi_fn, index=False, lineterminator='\n')
 
    # create the targets arrays
    raw_train_target = mergedInfo.loc[mergedInfo['dataset_ID']==1] # group 1 = training set
    raw_validation_target = mergedInfo.loc[mergedInfo['dataset_ID']==2] # group  2 = validation set
    raw_test_target = mergedInfo.loc[mergedInfo['dataset_ID']==3] # group 3 = test set
    print("raw_train_target df is this long",len(raw_train_target.index))
    print("raw_validation_target df is this long",len(raw_validation_target.index))
    print("raw_test_target df is this long",len(raw_test_target.index))
 
    train_target = raw_train_target['class_label']
    validation_target = raw_validation_target['class_label']
    test_target = raw_test_target['class_label']
    print("train_target df is this long",len(train_target.index))
    print("validation_target df is this long",len(validation_target.index))
    print("test_target df is this long",len(test_target.index))
 
    train_target = np.array(train_target)
    validation_target = np.array(validation_target)
    test_target = np.array(test_target)
    print("train_target array is this long",len(train_target))
    print("validation_target array is this long",len(validation_target))
    print("test_target array is this long",len(test_target))
 
    #===========================================================================================#
    #BUILD A TIME SERIES CLASSIFICATION MODEL AND A SINGLE LAYER LSTM MODEL
    #===========================================================================================#
 
   
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(num_samples, num_channels)))
 
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
   
    adam = Adam(learning_rate=lr)
    chk = ModelCheckpoint(best_model_pkl, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
   
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
   
    history = model.fit(train, train_target, epochs=epochs, batch_size=batch_size, callbacks=[chk], validation_data=(validation,validation_target))
    print("*** Model Summary ***")
 
    def printsummary(s):
        modelsummary_file = os.path.join(experimental_dir, 'model_summaries.txt')
        with open(modelsummary_file,'a') as f:
            print(s, file=f)
        f.close()
 
    print(model.summary())
    model.summary(print_fn=printsummary)
 
    # summarize data for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    model_file = os.path.join(experimental_dir, timestamp + '_model_plot_1' + '.png')
    plt.savefig(model_file)
    plt.close()
    #plt.show()    # this opens a popup of the accuracy if you wish to see it, but
                   # it is also stored in the experiments folder
 
    #summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    model_file = os.path.join(experimental_dir, timestamp + '_model_plot_2' + '.png')
    plt.savefig(model_file)
    plt.close()
    #plt.show()    # this opens a popup of the loss if you wish to see it, but
                   # it is also stored in the experiments folder
 
    print("length of test is", len(test))
    test_preds = (model.predict(test) > 0.5).astype("int32")
    print("length of test_preds is", len(test_preds))
 
    accuracy= accuracy_score(test_target, test_preds)
    print("Accuracy Score: ", accuracy)
 
    ### Save the results to the experiment directory for future reference
    modeldiagram_file = os.path.join(experimental_dir, timestamp + '_model_plot' +'.png')
    backup_groupsFile = os.path.join(experimental_dir, timestamp + os.path.basename(groupsFile))
    backup_targetsFile = os.path.join(experimental_dir, timestamp + os.path.basename(targetsFile))
    shutil.copy(groupsFile, backup_groupsFile)
    shutil.copy(targetsFile, backup_targetsFile)
 
    plot_model(model, to_file=modeldiagram_file, show_shapes=True, show_layer_names=True)
 
    experimentdata_file = os.path.join(experimental_dir,'experimentdata.txt')
    experimentdata_fn = open(experimentdata_file, "a")
    spacedchannels=' '.join(channels)
    logline=str(timestamp)+', '+str(accuracy)+', '+datasubset+', '+spacedchannels+', '+str(lr)+', '+str(lstm_units)+', '+str(epochs)+', '+str(batch_size)+'\n'
    experimentdata_fn.writelines(logline)
    experimentdata_fn.close()
 
exit()
