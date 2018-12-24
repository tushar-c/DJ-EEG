import scipy.io as sio 
import os
import numpy as np


def init_train_files(extension=".mat"):
    train_files = [f for f in os.listdir() if os.path.splitext(f)[1] == extension]
    print("{} {} files found".format(len(train_files), extension))
    print('files found:', train_files)
    return train_files


def get_runs_data(train_files):
    print('parsing train files...')
    runs_data = []
    for t in train_files:
        print('Fetching data from {}'.format(t))
        loader = sio.loadmat(t)
        for keys, _ in loader.items():
            if keys == 'data':
                data = loader[keys]
                for runs in data:
                    for run in range(len(runs)):
                        if run > 2: # first 3 are calibration runs
                            current_run = runs[run]
                            true_data = current_run[0][0]
                            each_run = []
                            for field in true_data:
                                each_run.append(field)
                            runs_data.append(each_run)
    return runs_data
        
     
def extract_data(runs_data):
    print('finalizing data...')
    raw_eeg_data = []

    for r in runs_data:
        # labelling based on manual inspection
        run_eeg = r[0]
        run_time_points = r[1]
        run_labels = r[2]
        run_sample_rate = r[3]
        run_string_labels = r[4]
        run_artifacts = r[5]
        run_gender = r[6]
        run_age = r[7]

        sliced_eeg = []

        step = 0
        for tp in range(len(run_time_points)):
            t = run_time_points[tp][0]
            eeg_slice = run_eeg[step: t]
            label = np.array([run_labels[tp][0]])
            label_string = run_string_labels[0][label - 1]
            # final data available in format : (X, y, direction(y), fs, artifacts, gender, age)
            # direction is not an actual function, just a helping abstraction to describe the int -> string mapping of directions 
            train_data = [eeg_slice, label, label_string, run_sample_rate, run_artifacts, run_gender, run_age]
            sliced_eeg.append(train_data)
            step = t        
        raw_eeg_data.append(sliced_eeg)
    
    return raw_eeg_data


def flat_and_pad(data):
    flattened_data = [v.ravel() for v in data]
    target_length = max([j.size for j in flattened_data])
    N = len(flattened_data)
    for ev in range(N):
        zero_vec_len = int(target_length - len(flattened_data[ev]))
        zero_vec = np.zeros(zero_vec_len)
        data[ev] = np.concatenate((flattened_data[ev], zero_vec))
    return data


def get_raw_eeg(data):
    eeg_data = []
    for each_run in data:
        for fields in each_run:
            eeg_data.append(fields[0])    
    return eeg_data


def transform_to_padded(data, padded_eeg):
    for each_run in data:
        for fields in each_run:
            fields[0] = padded_eeg[0]
    return data
