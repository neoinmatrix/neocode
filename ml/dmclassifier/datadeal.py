# coding=utf-8
import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import  metrics

class DataDeal:
    def getPima(self):
        train=pd.read_csv("./data/pima.csv",header=None)
        train_normal=(train-train.min())/(train.max()-train.min())
        target=np.array(train[8].values,dtype=np.int)
        train_normal=train_normal.drop(8,axis=1)
        data=np.array(train_normal.values,dtype=np.float)
        nums=len(data[:,0]) # pima has x numbers items
        types=2 # pima has two types 1 2
        return  data,target,nums,types
    def getIris(self):
        train=pd.read_csv("./data/iris.csv",header=None)
        train_normal=(train-train.min())/(train.max()-train.min())
        target=np.array(train[4].values,dtype=np.int)
        train_normal=train_normal.drop(4,axis=1)
        data=np.array(train_normal.values,dtype=np.float)
        nums=len(data[:,0]) # iris has x numbers items
        types=3 # iris has two types 1 2 3
        return data,target,nums,types
    def test(self,clf,data,target,nums,types,isprint=False):
        kf = KFold(n_splits=10, shuffle=True,random_state=np.random.randint(11))
        accuracy=0.0
        confusion=np.zeros([types,types])
        for train_index, test_index in kf.split(range(nums)):
            X=data[train_index]
            y=target[train_index]
            clf.fit(X,y)
            X_test=data[test_index]
            y_expected = target[test_index]
            predicted = clf.predict(X_test)
            conf_tmp=metrics.confusion_matrix(y_expected, predicted)
            confusion+=conf_tmp
            accy_tmp=metrics.accuracy_score(y_expected, predicted)
            accuracy+=accy_tmp
            if isprint:
                print accy_tmp
        # p,r,t = metrics.precision_recall_curve(y_expected, predicted)
        return [confusion,accuracy/10.0]  
    def saveResult(self,data):
        pass
        pass
    def calcPRCurve(self,data):
        pass
        pass
if __name__=="__main__": 
    dd=DataDeal()
    # iris=dd.getIris()
    # data,target,nums,types=iris
    pima=dd.getPima()
    data,target,nums,types=pima
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    result=dd.test(clf,data,target,nums,types)
    print "=========pima", result[1]