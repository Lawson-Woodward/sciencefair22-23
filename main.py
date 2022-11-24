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


#created the processing directory:
#create_data_processing_dir(processed_data_dir)

#extracted each subject to the processing direcotry
#extract_subjects_to_dir(full_data_dir, processed_data_dir)

#extracted all of the subject files for each subject the processing directory:
#for file in pathlib.Path(processed_data_dir).glob("co*"):
#    print ("file is %s" % file)
#    extract_files_to_subject(processed_data_dir, file)
