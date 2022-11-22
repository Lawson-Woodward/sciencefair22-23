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
 
# Create the processing directory where any data manipulation will happen
def create_data_processing_dir(processed_data_dir):
    try:
        os.makedirs(processed_data_dir, exist_ok = True)
        print ("Target dir '%s' for processed data successfully created" % processed_data_dir)
    except OSError as error:
        print ("Directory '%s' cannot be created" % processed_data_dir)

def extract_to_dir(full_dir, processed_dir):
    for file in pathlib.Path(full_dir).glob('*.tar.gz'):
        input = tarfile.open(file)
        input.extractall(processed_dir)
        input.close()

#create_data_processing_dir(processed_data_dir)
extract_to_dir(full_data_dir, processed_data_dir)
