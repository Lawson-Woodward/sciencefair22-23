import os

root_dir = 'C:/eeg/sciencefair22-23/data1/eeg_full/experiments'
for i in range(19, 85):
    experiment_dir = os.path.join(root_dir, 'Experiment {}'.format(i))
    os.makedirs(experiment_dir)