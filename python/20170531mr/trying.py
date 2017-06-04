# coding=utf-8
import dataset
import datadeal
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import  metrics

# this is demo about 1.how to get data 2.how test model
def demotest():
    # test data 
    ds=dataset.DataSet()
    ds.getTrainData()
    dt=datadeal.DataTrain()
    clf = SVC(C=1)
    X=ds.getPosOfMouse(0)
    y=ds.train["labels"]
    dt.trainTest(clf,X,y)

# use the start mouse position to predict with ann
def getReulst1():
    ds=dataset.DataSet()
    ds.getTrainData()
    dt=datadeal.DataTrain()
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10), random_state=1)
    y=ds.train["labels"]
    mouses=ds.train["mouses"]
    X=[]
    for i in range(ds.train["size"]):
        xs=mouses[i][0]
        ys=mouses[i][1]
        X.append([xs[0],ys[0]])
    X=np.mat(X)
    # dt.trainTest(clf,X,y)
    dt.train(clf,X,y)
    def f(idx,mouse,goal,label):
        if idx==False:
            return False
        xarr=mouse[0]
        yarr=mouse[1]
        return np.array([xarr[0],yarr[0]]).reshape([1,2])
    dt.testResultAll(ds,f,savepath='./data/ann_mouse_start.txt')

# use the start mouse position to predict with svm
def getResult2():
    ds=dataset.DataSet()
    ds.getTrainData()
    dt=datadeal.DataTrain()
    clf = SVC(C=1.5)
    y=ds.train["labels"]
    mouses=ds.train["mouses"]
    X=[]
    for i in range(ds.train["size"]):
        xs=mouses[i][0]
        ys=mouses[i][1]
        X.append([xs[0],ys[0]])
    X=np.mat(X)
    # dt.trainTest(clf,X,y)
    dt.train(clf,X,y)
    def f(idx,mouse,goal,label):
        if idx==False:
            return False
        xarr=mouse[0]
        yarr=mouse[1]
        return np.array([xarr[0],yarr[0]]).reshape([1,2])
    dt.testResultAll(ds,f,savepath='./data/svm_mouse_start.txt')

# use the goal mouse position to predit with ann
def getReuslt3():
    ds=dataset.DataSet()
    ds.getTrainData()
    dt=datadeal.DataTrain()
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(20), random_state=1)
    y=ds.train["labels"]
    X=ds.train["goals"]
    X=np.mat(X)
    # dt.trainTest(clf,X,y)
    dt.train(clf,X,y)
    def f(idx,mouse,goal,label):
        if idx==False:
            return False
        return np.array(goal).reshape([1,2])
    dt.testResultAll(ds,f,savepath='./data/ann_goal.txt')

if __name__=="__main__":
    demotest()
    pass