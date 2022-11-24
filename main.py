import pandas as pd
import os
import pathlib
import tarfile
import gzip
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
 
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

full_data_dir = 'C:/eeg/sciencefair22-23/data/eeg_full/'
processing_dir = 'processed'
processed_data_dir = os.path.join(full_data_dir, processing_dir)

channels = ['AF1', 'AF2','AF7','AF8','AFZ','FP1', 'FP2', 'CPZ', 'CZ', 'FCZ','FPZ', 'FT7', 'FT8', 'FZ','O1','O2','OZ','POZ','PZ','PO1','PO2','PO7','PO8', 'S1', 'T7','T8','TP7','TP8'] + [f'F{n:01}' for n in range(1,9)] + [f'C{n:01}' for n in range(1,8)] + [f'CP{n:01}' for n in range(1,7)] + [f'F{n:01}' for n in range(1,6)]  + [f'FC{n:01}' for n in range(1,7)] + [f'P{n:01}' for n in range(1,9)]

num_channels = 68
num_samples = 256

#creates the processing directory where data manipulation will happen
def create_data_processing_dir(processed_data_dir):
    try:
        os.makedirs(processed_data_dir, exist_ok = True)
        print ("Target dir '%s' for processed data successfully created" % processed_data_dir)
    except OSError as error:
        print ("Directory '%s' cannot be created" % processed_data_dir)

#extracts subject files in the data folder
def extract_subjects_to_dir(full_dir, processed_dir):   
    for file in pathlib.Path(full_dir).glob('*.tar.gz'):
        #input = tarfile.open(file)
        #input.extractall(processed_dir)
        #input.close()
        subject = tarfile.open(file, "r:gz") 
        subject.extractall(processed_dir)
        subject.close()

#extracts files in the subject files
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
    print(file_data)

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
    print(file_data)

    dirname = os.path.dirname(os.path.abspath(file_name))
    fname = os.path.basename(file_name)
    new_fname = os.path.join(dirname, 'new_' + fname)
    my_df = pd.DataFrame(file_data)
    my_df.to_csv(new_fname, header=channels, index=False, lineterminator='\n')
    return new_fname


#created the processing directory:
#create_data_processing_dir(processed_data_dir)

#extracted each subject to the processing direcotry
#extract_subjects_to_dir(full_data_dir, processed_data_dir)

#extracted all of the subject files for each subject the processing directory:
#for file in pathlib.Path(processed_data_dir).glob("co*"):
#    print ("file is %s" % file)
#    extract_files_to_subject(processed_data_dir, file)

create_array("C:/eeg/sciencefair22-23/data/eeg_full/processed/co2a0000364/co2a0000364.rd.000", channels)

