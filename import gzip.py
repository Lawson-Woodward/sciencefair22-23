import gzip
import pickle
import pandas as pd

f = pd.read_pickle('C:/eeg/sciencefair22-23/data/eeg_full/experiments/experiment9/20230115122302_best_model.pkl', compression='infer', storage_options=None)  
print(f)