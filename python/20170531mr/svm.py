# coding=utf-8

import datadeal as dd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn import  metrics
from sklearn.svm import SVC

def getSingle(mouse,gn=10):
    feature=np.zeros([gn],dtype=np.float)
    idxm= len(mouse)-1 if len(mouse)>1 else 1
    idx=np.random.randint(1,idxm,size=[gn])
    idx=np.sort(idx,axis=-1)
    if len(mouse[0])<2:
        return np.array([0.0]*gn)
    for i in range(gn):
        pos=idx[i]
        ex=mouse[0][pos]
        ey=mouse[1][pos]
        ez=mouse[2][pos]
        sx=mouse[0][pos-1]
        sy=mouse[1][pos-1]
        sz=mouse[2][pos-1]
        dt=0.0
        dt+=(ex-sx)**2
        dt+=(ey-sy)**2
        dt+=(ez-sz)**2
        dt=dt**0.5
        feature[i]=dt
    return feature

def getFeature(idx,gn=10):
    n=len(idx)
    features=np.array([0.0]*gn)
    for i in idx:
        row=getSingle(dd.mouses[i],gn=gn)
        features=np.row_stack([features,row])
    return features[1:]

def trainset():
    
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10,10), random_state=1)
    X=getFeature([i for i in range(len(dd.mouses))],gn=10)
    y=np.array(dd.labels)
    clf.fit(X,y.ravel())
    print "train over!"
    return clf

if __name__=="__main__y":
    clf=trainset()

if __name__=="__main__":
    dd.initdata()
    xarr=[]
    yarr=[]
    for v in dd.goals:
        arr=v.split(',')
        xarr.append(float(arr[0]))
        yarr.append(float(arr[1]))
    xarr=np.array(xarr).reshape([3000,1])
    yarr=np.array(yarr).reshape([3000,1])
    # print xarr
    # clf = SVC(C=1.5, kernel='linear' )

    kf = KFold(n_splits=10, shuffle=True,random_state=np.random.randint(3000))
    # clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10,10), random_state=1)
    clf = SVC(C=1.5)
    accuracy=0.0
    for train_index, test_index in kf.split(range(len(dd.mouses))):
        X=xarr[train_index]
        # print "get train"
        y=np.array(dd.labels[train_index])
        X_test=xarr[test_index]
        # print "get test"
        expected=dd.labels[test_index]
        clf.fit(X,y.ravel())
        # print "get fit"
        predicted = clf.predict(X_test)
        # print "get predit"
        accy_tmp=metrics.accuracy_score(expected, predicted)
        accuracy+=accy_tmp
        print "get predit rate:%f"%accy_tmp
        # # break
    print accuracy/10.0
