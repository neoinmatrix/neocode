# coding=utf-8
import dataset
import datadeal
import datadraw
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import  metrics
import matplotlib.pyplot as plt 

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

# use the goal mouse y mse and x vector msc to predit with ann
def getReuslt_mse():
    ds=dataset.DataSet()
    ds.getTrainData()
    dt=datadeal.DataTrain()
    mouses=ds.train["mouses"]
    labels=ds.train["labels"]
    def getReuslt_mse_getdata(mouses):
        mse=[]
        for v in mouses:
            mse.append([v[0][0],v[1][0],np.std(v[1]),np.std(v[2])])
        return np.array(mse)
    mse=getReuslt_mse_getdata(mouses)
    # drawing picture to analyst ==================================
    # dw=datadraw.DataDraw(typex='3d')
    # print goals.shape
    # dw.drawgoal([mse[:2600,0],mse[:2600,1]],'b')
    # dw.drawgoal([mse[2600:,0],mse[2600:,1]],'r')
    # for i in range(2600):
    #   dw.draw3dgoal(mse[i],c='b')
    # for i in range(2600,3000):
    #   dw.draw3dgoal(mse[i],c='r')
    # plt.show()
    # dw.drawbatchgoal(,'r')
    # drawing picture to analyst ==================================
    # clf=SVC(C=1)
    clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(40), random_state=1)
    y=ds.train["labels"]
    X=mse
    X=np.mat(X)
    dt.trainTest(clf,X,y)
    exit()
    dt.train(clf,X,y)
    def f(idx,mouse,goal,label):
        if idx==False:
            return False
        tmp=[mouse[0][0],mouse[1][0],np.std(mouse[1]),np.std(mouse[2])]
        return np.array(tmp).reshape([1,4])
    dt.testResultAll(ds,f,savepath='./data/xytmse0605.txt')

# calc the right item number
def calcRightNumbers():
    print datadeal.calcScore(15268.0,15268.0)
    print datadeal.calcScoreRerve(0.7245,16776.0)
    print datadeal.calcScoreRerve(0.7508,20561.0)
    # print datadeal.calcScoreRerve(0.2941,100000.0)
# draw the PR rate map
def drawPRmap():
    x=[]
    labels=[]
    for i in range(10000,25000,1000):
        rm=i+1000
        labels.append(rm/100)
        for j in range(10000,25000,1000):
            jm=j+1000
            if rm>jm:
                tmp=0.0
            else:
                tmp=datadeal.calcScore(float(rm),float(jm))
            # x.append("%.2f"%tmp)
            x.append(tmp)
    r=np.array(x).reshape(15,15)
    # print r
    # labels=[]
    # for j in range(10000,25000,1000):
    #     labels.append(j)
    # labels=np.array(labels)
    datadraw.plot_confusion_matrix(r,labels,'a','b',1)

if __name__=="__main__":
    # demotest()
    # getReuslt_mse()
    # calcRightNumbers()
    drawPRmap()
    pass