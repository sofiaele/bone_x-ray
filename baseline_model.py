from numpy import random
import numpy
import sklearn.metrics
import pandas as pd
from sklearn.dummy import DummyClassifier


def baseline(data_path):
    df = pd.read_csv(data_path)
    labels_distribution_train = {"0": 0, "1": 0}
    for index, row in df.iterrows():
        label = str(row['label'])
        labels_distribution_train[label] += 1
    n_0 = labels_distribution_train['0']
    n_1 = labels_distribution_train['1']
    print(n_0)
    print(n_1)
    p_0 = n_0 / (n_0 + n_1)
    p_1 = n_1 / (n_0 + n_1)
    y = random.choice([0, 1], p=[p_0, p_1], size=(3197))
    y_p = numpy.zeros((3197, 1))
    print(sklearn.metrics.f1_score(y, y_p, average='macro'))

    ############## second way #################
    X_train = df['path'].values.tolist()
    y_train = df['label'].values.tolist()
    dummy = DummyClassifier(strategy='most_frequent')  # or 'stratified', 'uniform', etc.
    dummy = dummy.fit(X_train, y_train)  # X_train is the training features, y_train is the corresponding labels
    test = pd.read_csv('utils/test.csv')
    X_test = test['path'].values.tolist()
    y_test = test['label'].values.tolist()
    y_pred = dummy.predict(X_test)  # X_test is the testing features
    print(sklearn.metrics.f1_score(y_test, y_pred, average='macro')) # y_test is the true labels for the testing data


baseline('utils/train.csv')