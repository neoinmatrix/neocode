import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

# data = load_iris()

# features = data['data']
# target = data['target']
# print features
# print [target==1]
# print features[target==1,0]
# feature_names = data['feature_names']

# print type(features)
# exit()
import pandas as pd
data=pd.read_csv("./a.csv",header=None)
features =data[:][[i for i in range(8)]].values
# features =data[:][[0,6]].values
target = data[:][8]
labels=["nopima","pima"]
feature_names=[i for i in range(10)]
# target=target.reshape(1,len(target))[0]
# print target
# exit()

# print [target.T==1]
# print target.reshape(1,len(target))
# print features
for i in range(len(features[0,:])):
    min=features[:,i].min()
    max=features[:,i].max()
    # print features[:,i].mean(),features[:,i].std()
    for j in range(len(features[:,i])):
        features[j,i]=(features[j,i]-min)/(max-min)
# print len(features[target == 1,0])
# print len(features[target == 2,0])
# for i in [target==1]:
#     print target[i]

# print features[target==1,0]

# exit()
# print feature_names
pairs=[(i,8) for i in range(8)]
# print pairs
# exit()

# pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

for i,(p0,p1) in enumerate(pairs):
    plt.subplot(4,2,i+1)
    marker="o"
    c="g"
    plt.scatter(features[:,p0], target, marker=marker, c=c)
    # for t,marker,c in zip(range(1,3),">o","rg"):
    #     plt.scatter(features[target == t,p0], features[target == t,p1], marker=marker, c=c)
    plt.xlabel(feature_names[p0])
    plt.ylabel(feature_names[p1])
    # plt.xticks([])
    # plt.yticks([])
# plt.savefig('./1400_02_01.png')
plt.show()
