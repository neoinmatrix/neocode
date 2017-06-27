import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

train=pd.read_csv("./pima.csv",header=None)
train_normal=(train-train.mean())/(train.max()-train.min())
train_normal[8]=train[8]

features =train_normal[:][[i for i in range(8)]].values
target = train_normal[:][8]
labels=["nopima","pima"]
feature_names=[i for i in range(9)]
colors=['r','y','g','b']
pairs=[(i,8) for i in range(8)]
for i,(p0,p1) in enumerate(pairs):
    # print i,(p0,p1)
    plt.subplot(4,2,i+1)
    marker="."
    c=colors[i%4]
    plt.scatter(features[:,p0], target, marker=marker, c=c)
    plt.xlabel(feature_names[p0])
    plt.ylabel(p0)
# plt.show()

ref_matrix=np.zeros([9,9])
for i in range(9):
    for j in range(9):
        ref_matrix[i][j]=train[i].corr(train[j])
# print ref_matrix
from utils import *
plot_confusion_matrix2(ref_matrix,[i for i in range(9)],'Correlation (reflect rate between two features)','Correlation (reflect rate between two features)')

