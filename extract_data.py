import os
import scipy.io as sio 
import numpy as np


def init_train_files(extension=".mat", folder_name='EegData'):
    
    '''
    Make sure the data folder is contained in the current directory.
    Fetch the data by checking file extenstions against the arg 'extension'

    Parameters: (extension, folder_name)
    * extension: type = string; refers to extension of data files
    * folder_name: type = string; gives name of the folder containing the files

    Return: (train_files) 
    * data files of file type = extension
    '''

    # we want the data folder to be in the current directory
    assert folder_name in os.listdir()
    data_path = os.path.join(os.path.abspath(os.curdir), folder_name)

    os.chdir(data_path) # go in the data folder

    train_files = [f for f in os.listdir() if os.path.splitext(f)[1] == extension]
    
    print("{} {} files found".format(len(train_files), extension))
    print('files found:', train_files)
    
    return train_files


def get_runs_data(train_files):
    
    '''
    The data as provided in the files contains a lot of nesting.
    This function extracts and puts it into a more usable form. As -
    Some runs are not meant to be used as those in which the subject was-
    actively training. Rather, those are calibration runs.

    Parameters: (train_files)
    * train_files: type = list; refers to the training (data) files as obtained above

    Return: (runs_data)
    * runs_data : list containing the data from all the runs as numpy arrays
    '''

    print('parsing train files...')
    runs_data = []
    for t in train_files:
        print('Fetching data from {}'.format(t))
        loader = sio.loadmat(t)
        for keys, _ in loader.items():
            if keys == 'data': # we only want the data 
                data = loader[keys]
                for runs in data:
                    for run in range(len(runs)):
                        if run > 2: # first 3 runs are calibration runs
                            current_run = runs[run]
                            true_data = current_run[0][0]
                            each_run = []
                            for field in true_data:
                                each_run.append(field)
                            runs_data.append(each_run)
    return runs_data
        
     
def extract_data(runs_data):
    
    '''
    Extract the individual fields from the data, format them properly.
    Most importantly, divide the data into segments based on the imagery -
    that was being performed during the time-window in which the data was -
    recorded.

    Parameters: (runs_data)
    * runs_data: list containing the data as numpy arrays 

    Return: (raw_eeg_data)
    * raw_eeg_data: list containing the time-separated eeg recordings-
                    that can be used for further analysis. Each recording has-
                    dimensions = (time points, channels). Note the format -
                    (n_rows, n_cols).
    '''

    print('finalizing data...')
    raw_eeg_data = []

    for r in runs_data:
        # labelling based on manual inspection
        run_eeg = r[0].T
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
            train_data = [eeg_slice, label, label_string, run_sample_rate, run_artifacts, run_gender, run_age]
            sliced_eeg.append(train_data)
            step = t        

        raw_eeg_data.append(sliced_eeg)
    
    return raw_eeg_data


def flat_and_pad(data):
    
    '''
    To act as input to an ML Model (eg: ANNs) we need uniform-sized inputs.
    This function does the trivial zero padding by adding zeros to each vector.
    The number of zeros to be added is calculated so that all vectors have the-
    same size as the largest vector, i.e., one with the most number of elements.

    Parameters: (data)
    * data: the raw eeg data, usually obtained as matrices with shape-
            (time_points, channels)
    '''
    
    flattened_data = [v.ravel() for v in data]
    target_length = max([j.size for j in flattened_data])
    N = len(flattened_data)
    for ev in range(N):
        zero_vec_len = int(target_length - len(flattened_data[ev]))
        zero_vec = np.zeros(zero_vec_len)
        data[ev] = np.concatenate((flattened_data[ev], zero_vec))
    return data


def get_raw_eeg(data):
    '''
    A helper function to return just the raw eeg data from the whole dataset

    Parameters: (data)
    * data: the whole dataset

    Return: (eeg_data)
    * depending on previous processing, might be uniform-
    or non-uniform (matrices or vectors).
    '''

    eeg_data = []
    for each_run in data:
        for fields in each_run:
            eeg_data.append(fields[0])    
    return eeg_data


def transform_to_padded(data, padded_eeg):
    '''
    To be used after zero padding. This function simply takes the padded-
    data and replaces the original non-uniform matrices in the dataset with-
    the new zero-padded vectors

    Parameters: (data, padded_eeg)
    * data: original data
    * padded_eeg: the eeg data after being zero padded to be made uniform

    Return: (data)
    * data = the dataset with eeg vectors of uniform length

    '''
    for each_run in data:
        for fields in each_run:
            fields[0] = padded_eeg[0]
    return data
