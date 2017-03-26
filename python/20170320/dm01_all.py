# coding=utf-8
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from dm01_oth import *
from utils import *

data = load_iris()
features = data['data']
# labels = data['feature_names']
target = data['target']
labels = data['target_names']
# exit()
# using the sklearn module 
predicts=np.array([0 for i in range(9)]).reshape(3,3)
kf = KFold(n_splits=10, random_state=True)
mlpc = MLPClassifier()
for train_index, test_index in kf.split(features):
    mlpc.fit(features[train_index][:,[1,2,3]],target[train_index])
    result=mlpc.predict(features[test_index][:,[1,2,3]])
    for i,j in zip(result,target[test_index]):
        predicts[i,j]+=1
print predicts
right=0
for i in range(len(predicts[0])):
    right+=predicts[i,i]
print float(right)/float(np.sum(predicts))
plot_confusion_matrix(predicts,labels,"ann_iris","confusion matrix of ann predict of iris")
exit()

# coding by self ======================================
predicts=np.array([0 for i in range(9)]).reshape(3,3)
kf = KFold(n_splits=10, random_state=True)
Config.nn_input_dim=3
Config.nn_output_dim=3
for train_index, test_index in kf.split(features):
    X=features[train_index][:,[1,2,3]]
    y=target[train_index]
    model = fit_model(X, y, 8,num_passes=800, print_loss=False)
    X_t=features[test_index][:,[1,2,3]]
    result= predict(model,X_t)
    for i,j in zip(result,target[test_index]):
        predicts[i,j]+=1
print predicts
right=0
for i in range(len(predicts[0])):
    right+=predicts[i,i]
print float(right)/float(np.sum(predicts))