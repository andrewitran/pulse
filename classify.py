import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train():
    predictors_temp = open('predictors.csv')
    predictors = csv.reader(predictors_temp, delimiter=',')
    predictors = np.array(list(predictors)).astype(np.float)
    predictors_temp.close()

    labels_temp = open('labels.csv')
    labels = csv.reader(labels_temp, delimiter=',')
    labels = np.array(list(labels))
    labels = labels.flatten()
    labels_temp.close()

    randomforest = RandomForestClassifier(n_estimators=20)
    trained = randomforest.fit(predictors, labels)

    return trained

def test(trained, data):
    result = trained.predict(data)

    return result
