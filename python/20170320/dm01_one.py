# coding=utf-8
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

data = load_iris()
features = data['data']
labels = data['feature_names']
target = data['target']

index_train=np.random.randint(150,size=20)
index_test=np.random.randint(150,size=20)
# print index_train
# print index_test
X=features[index_train]
y=target[index_train]
# print X
# print y
X_t=features[index_test]
y_t=target[index_test]

mlpc = MLPClassifier().fit(X,y)
result=mlpc.predict(X_t)  
print result
print y_t