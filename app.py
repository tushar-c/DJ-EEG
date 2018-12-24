import numpy as np 
import bci 
import process

# set save file name
save_file_name = 'MI_DATA'

# get train files
train_files = bci.init_train_files()

# get data for all runs 
runs_data = bci.get_runs_data(train_files)

# extract into more usable form
data = bci.extract_data(runs_data)

# get raw eeg for uniformalizing data
raw_eeg = bci.get_raw_eeg(data)

# pad for uniformity
padded_eeg = bci.flat_and_pad(raw_eeg)

# finally get the train data
padded_data = bci.transform_to_padded(data, padded_eeg)

# save the data
np.save(save_file_name, np.array(padded_data))

# make sure everything worked
print('Data Extracted and Processed, File Saved to Disk in the Current Directory...')

# train a logistic regression model
features, labels = process.load_data()

# reformat for sklearn
features = np.array(features)
labels = np.array(labels).reshape(len(labels), )

print('Train Data Loaded, Now Training...')

# score the trained model for training accuracy and make predictions
train_score, preds = process.train_logistic_regression(features, labels)

# again, make sure it all works
print('Training Complete, Score on train data = {}'.format(train_score))

# train svm
print('Training SVM...') 
svm = process.train_svm(features, labels)
print('SVM score = {}'.format(svm))

