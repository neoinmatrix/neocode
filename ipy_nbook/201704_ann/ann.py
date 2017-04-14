# coding=utf-8
import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn import  metrics

train=pd.read_csv("./pima.csv",header=None)
train_normal=(train-train.mean())/(train.max()-train.min())
train_normal[8]=train[8]

kf = KFold(n_splits=10, shuffle=True,random_state=np.random.randint(11))
clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(50,), random_state=1)
accuracy=0.0
confusion=np.zeros([2,2])
for train_index, test_index in kf.split(train_normal.index):
    X=train_normal.ix[train_index].drop(8,axis=1).values
    y=train_normal.ix[train_index][8].values
    clf.fit(X,y)
    X_test=train_normal.ix[test_index].drop(8,axis=1).values
    predicted = clf.predict(X_test)
    expected = train_normal.ix[test_index][8].values
    conf_tmp=metrics.confusion_matrix(expected, predicted)
    confusion+=conf_tmp
    accy_tmp=metrics.accuracy_score(expected, predicted)
    accuracy+=accy_tmp
    print conf_tmp
    print accy_tmp
print confusion
print accuracy/10.0
