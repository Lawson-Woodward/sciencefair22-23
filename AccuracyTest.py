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
 
full_data_dir = 'C:/eeg/sciencefair22-23/data/eeg_full/'  
 
processing_dir = 'processed' #manually clean up the new_ files in this dir by searching for name:new_* and dleteing
processed_data_dir = os.path.join(full_data_dir, processing_dir) #Creating a new file path called, C:/eeg/sciencefair22-23/data/eeg_full/processed
experimental_dir = os.path.join(full_data_dir, 'experiments/')
groupsFile =  os.path.join(processed_data_dir,'groups.csv')
targetsFile = os.path.join(processed_data_dir,'targets.csv')
 
 
 
#now = datetime.now()
#timestamp = now.strftime("%Y%m%d%H%M")
#best_model_pkl = os.path.join(experimental_dir, timestamp+'_'+'best_model.pkl')
 
#Important for Machine Learning, specifes paramaters
channels = ['FP1', 'FP2', 'FPZ']
#channels = ['AF1', 'AF2','AF7','AF8','AFZ','FP1', 'FP2', 'CPZ', 'CZ', 'FCZ','FPZ', 'FT7', 'FT8', 'FZ','O1','O2','OZ','POZ','PZ','PO1','PO2','PO7','PO8', 'T7','T8','TP7','TP8'] + [f'F{n:01}' for n in range(1,9)] + [f'C{n:01}' for n in range(1,7)] + [f'CP{n:01}' for n in range(1,7)] + [f'FC{n:01}' for n in range(1,7)] + [f'P{n:01}' for n in range(1,9)]
 
num_channels = len(channels)
num_samples = 256
 
#subject_dirs = pathlib.Path("C:\eeg\sciencefair22-23\data\eeg_full\processed").glob("co*") #Finding all files that starts with co
subject_dirs=list(pathlib.Path(processed_data_dir).glob("co*")) #Finding all files that starts with co
 
# Experimental Values
datasubset='S1 obj' # S is for all, 'S1 obj', 'S2 match' or 'S2 nomatch'
lr=.0005
lstm_units=20
epochs= 10
batch_size=1
 
#===========================================================================================#
#WRITING FUNCTIONS THAT WE WILL LATER CALL
#===========================================================================================#
 
def create_data_processing_dir(processed_data_dir): #If no folder named processing, it creates one to store in the processed data later
    try:
        os.makedirs(processed_data_dir, exist_ok = True)
        print ("Target dir '%s' for processed data successfully created" % processed_data_dir)
    except OSError as error:
        print ("Directory '%s' cannot be created" % processed_data_dir)
 
def create_experimental_dir(experimental_dir): #If no folder named processing, it creates one to store in the processed data later
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
 
#processed dir > patients (co) > # of trials > list of electrodes
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
 
    create_file=0
    # Start by creating an empty 2d array that we can fill with data from a file
    data_array = np.zeros([num_samples, num_channels], dtype=np.float64)
 
    with open(filename) as f:
        content = f.readlines()
        if (datasubset in content[3]):
            create_file=1
            #for line in f:
            for line in content:
                newline = line.split()
                # Remove unknown channels. We were unable to determine what nd and Y stood for
                # since they are nonstandard channels.
                # We also need to exclude comments.
                if (newline[1] in ch_names and newline[0] != "#"):
                    index = ch_names.index(newline[1])
                    # print (index)
                    # load the data into the np array
                    data_array[int(newline[2]), int(index)] = newline[3]
        else:
            create_file=0
    f.close()
 
    if (create_file):
        dirname = os.path.dirname(os.path.abspath(filename))
        fname = os.path.basename(filename)
        newname = os.path.join(dirname,'new_' + fname)
        my_df = pd.DataFrame(data_array)
        my_df.to_csv(newname,header=ch_names, index=False, lineterminator='\n')  
        # This is our loaded data, transformed into an np array of channels rows x samples columns
        return newname
 
def read_trial_file(trial_file): #Getting the data
    trial_data_list = pd.read_csv(trial_file, sep='\s+', comment='#', header=0)
    return(trial_data_list)
 
def clean_subject_data_files(subject_dirs,datasubset): #Uses the above methods to actually move and change the data. Creates the new_co files
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
 
def create_groups_and_targets_file(subject_dirs): #Creates an A and C list, goes through each subject folder, and determined alchohol or control
 
    trial_files=[]
    for subject_dir in subject_dirs:
        fq_trial_files = pathlib.Path(subject_dir).glob("new*")
        for trial_file in fq_trial_files:
            base_fn = os.path.basename(trial_file)
            trial_files.append(base_fn)
   
    grouped_list=[]
    np.random.shuffle(trial_files)
    split_trial_files=np.split(trial_files,[int(len(trial_files)*0.6), int(len(trial_files)*0.8)])
    for i in [1,2,3]:
        for item in split_trial_files[i-1]:
            grouped_list.append([item,i])
   
    my_df = pd.DataFrame(np.array(grouped_list))
    groups_fn = os.path.join(processed_data_dir, 'groups.csv')
    my_df.to_csv(groups_fn, header=['#sequence_ID', 'dataset_ID'], index=False, lineterminator='\n')
 
    trial_files=[]
    targets=[]
 
    for subject_dir in subject_dirs:
        trial_files = pathlib.Path(subject_dir).glob("new*")
        a_or_c = os.path.basename(subject_dir)[3]
        if a_or_c == "a":
            a_or_c = "1"
        else:
            a_or_c = "0"
        for trial_file in trial_files:
            trial_fn = os.path.basename(trial_file)
            x = [trial_fn, a_or_c]
            targets.append(x)
    my_df = pd.DataFrame(targets)
    targets_fn = os.path.join(processed_data_dir, 'targets.csv')
    my_df.to_csv(targets_fn, header=['#sequence_ID', 'class_label'], index=False, lineterminator='\n')
 
#===========================================================================================#
#CALLING ALL OF THE FUNCTIONS THAT WERE WRITTEN SO FAR
#===========================================================================================#
 
for batch_size in (2, 4, 8, 16, 32, 64, 128):
    print ("batch_size is", batch_size)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    print("timestamp is", timestamp)
    best_model_pkl = os.path.join(experimental_dir, timestamp+'_'+'best_model.pkl')
 
    #created the processing directory:
    #create_data_processing_dir(processed_data_dir)
 
    #create directory to store experiment information
    #create_experimental_dir(experimental_dir)
   
    #extracted each subject to the processing directory
    #extract_subjects_to_dir(full_data_dir, processed_data_dir)
 
    #for file in pathlib.Path(processed_data_dir).glob("co*"):
    #   print ("file is %s" % file)
    #   extract_files_to_subject(processed_data_dir, file)
   
    #creates a cleaned up version of every single trial file in the subject files provided
    #clean_subject_data_files(subject_dirs,datasubset)
   
    #create_groups_and_targets_file(subject_dirs)
   
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
    model.add(Dense(1, activation="sigmoid"))
   
    adam = Adam(learning_rate=lr)
    chk = ModelCheckpoint(best_model_pkl, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
   
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
   
    history = model.fit(train, train_target, epochs=epochs, batch_size=batch_size, callbacks=[chk], validation_data=(validation,validation_target))
    #model.fit(train, train_target, epochs=200, batch_size=128, callbacks=[chk], validation_data=(validation,validation_target))
    ### 12/27 model.fit(train, train_target, epochs=1, batch_size=1, callbacks=[chk], validation_data=(validation,validation_target))
 
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
    #plt.show()
 
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
    #plt.show()
 
    #model = load_model(best_model_pkl)
    print("length of test is", len(test))
    test_preds = (model.predict(test) > 0.5).astype("int32")
    print("length of test_preds is", len(test_preds))
 
    accuracy= accuracy_score(test_target, test_preds) ####### confirm this is correct
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