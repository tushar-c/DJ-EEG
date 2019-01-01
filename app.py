import numpy as np 
import extract_data
import base

# set save file name
save_file_name = 'MI_DATA'

# get train files
train_files = extract_data.init_train_files()

# get data for all runs 
runs_data = extract_data.get_runs_data(train_files)

# extract into more usable form
data = extract_data.extract_data(runs_data)

# get raw eeg for uniformalizing data
raw_eeg = extract_data.get_raw_eeg(data)

# pad for uniformity
padded_eeg = extract_data.flat_and_pad(raw_eeg)

# finally get the train data
padded_data = extract_data.transform_to_padded(data, padded_eeg)

# save the data
np.save(save_file_name, np.array(padded_data))

# make sure everything worked
print('Data Extracted and processed, File Saved to Disk in the Current Directory...')

# train a logistic regression model
features, labels = base.load_data()

# reformat for sklearn
features = np.array(features)
labels = np.array(labels).reshape(len(labels), )

print('Train Data Loaded, Now Training...')

# score the trained model for training accuracy and make predictions
train_score, preds = base.train_logistic_regression(features, labels)

# again, make sure it all works
print('Training Complete, Score on train data = {}'.format(train_score))

# train svm
print('Training SVM...') 
svm = base.train_svm(features, labels)
print('SVM score = {}'.format(svm))

