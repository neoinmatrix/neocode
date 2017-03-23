# coding=utf-8
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

data = load_iris()
features = data['data']
labels = data['feature_names']
target = data['target']

predicts=np.array([0 for i in range(9)]).reshape(3,3)
kf = KFold(n_splits=10, random_state=True)
mlpc = MLPClassifier()
for train_index, test_index in kf.split(features):
    mlpc.fit(features[train_index][:,[1,2,3]],target[train_index])
    result=mlpc.predict(features[test_index][:,[1,2,3]])
    for i,j in zip(result,target[test_index]):
        predicts[i,j]+=1
print predicts
