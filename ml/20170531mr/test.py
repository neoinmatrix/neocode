# coding=utf-8
from sklearn import preprocessing
import numpy as np

X=np.array([1,2,3,4,5])
print X.mean()
print X.std()
print (X-X.mean)/X.std()
# print X.mean()
# print X.mean()
# X_scaled = preprocessing.scale(X)
# print X_scaled
# scaler = preprocessing.StandardScaler().fit(X)
# y=scaler.transform([5])
# print y