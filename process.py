import numpy as np
import sklearn 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC


def load_data(load_file_name = 'MI_DATA.npy'):
    data = np.load(load_file_name)

    features = []
    labels = []
    for runs in data:
        for run_data in runs:
            X, y = run_data[0], run_data[1]
            features.append(X)
            labels.append(y.ravel())
    return features, labels


def train_logistic_regression(features, labels, predict=True):
    features, labels = np.array(features), np.array(labels).reshape(len(labels), )
    reg = LogisticRegression().fit(features, labels)
    score = reg.score(features, labels)
    pred = 0
    if predict:
        pred = reg.predict(features)
    return score, pred


def train_svm(features, labels):
    clf = SVC()
    clf.fit(features, labels)
    score = clf.score(features, labels)
    return score


