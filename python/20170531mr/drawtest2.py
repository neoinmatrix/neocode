# coding=utf-8

import datadeal as dd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn import  metrics
from sklearn.svm import SVC

import matplotlib.pyplot as plt
def trainset():
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10,10), random_state=1)
    X=getFeature([i for i in range(len(dd.mouses))],gn=10)
    y=np.array(dd.labels)
    clf.fit(X,y.ravel())
    print "train over!"
    return clf


if __name__=="__main__":
    dd.initdata()
    xarr=[]
    yarr=[]
    # print dd.mouses[0]

    gxarr=[]
    gyarr=[]
    for v in dd.goals:
        arr=v.split(',')
        gxarr.append(float(arr[0]))
        gyarr.append(float(arr[1])) 

    for v in dd.mouses:
        # arr=v.split(',')
        xarr.append(float(v[0][0]))
        yarr.append(float(v[1][0]))
    # xarr=np.array(xarr).reshape([3000,1])
    # yarr=np.array(yarr).reshape([3000,1])

    for i in range(len(xarr)):
        gxarr[i]+=xarr[i]
        gyarr[i]+=yarr[i]

    # plt.scatter(xarr[:2600],yarr[:2600])
    # plt.scatter(xarr[2600:],yarr[2600:])

    plt.scatter(gxarr[:2600],gyarr[:2600])
    plt.scatter(gxarr[2600:],gyarr[2600:])

    plt.show()
    # dd.labels[test_index]
    # print xarr
    # clf = SVC(C=1.5, kernel='linear' )

    # kf = KFold(n_splits=10, shuffle=True,random_state=np.random.randint(3000))
    # # clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10,10), random_state=1)
    # clf = SVC(C=1.5)
    # accuracy=0.0
    # for train_index, test_index in kf.split(range(len(dd.mouses))):
    #     txarr=xarr[train_index]
    #     tyarr=yarr[train_index]
    #     X=np.array([txarr,tyarr])
    #     X=X.reshape([len(train_index),2])

    #     # print "get train"
    #     y=np.array(dd.labels[train_index])

    #     ttxarr=xarr[test_index]
    #     ttyarr=yarr[test_index]
    #     X_test=np.array([ttxarr,ttyarr])
    #     X_test=X_test.reshape([len(test_index),2])

    #     # print "get test"
    #     expected=dd.labels[test_index]
    #     clf.fit(X,y.ravel())
    #     # print "get fit"
    #     predicted = clf.predict(X_test)
    #     # print "get predit"
    #     accy_tmp=metrics.accuracy_score(expected, predicted)
    #     accuracy+=accy_tmp
    #     print "get predit rate:%f"%accy_tmp
    #     # # break
    # print accuracy/10.0
