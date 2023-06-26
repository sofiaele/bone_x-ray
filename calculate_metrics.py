from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from pandas import *
 
def metrics(path_to_voting_results):    
    data = read_csv(path_to_voting_results)

    y_pred = data['Prediction'].tolist()
    y_true = data['Truth'].tolist()


    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
