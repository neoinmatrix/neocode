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

def drawline():
    ds=dataset.DataSet()
    ds.getTrainData()
    dw=datadraw.DataDraw('2d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    # x,y,t=mouses[0]
    colors=dw.getColorsValue()
    for i in range(10):
        # if i==2:
        dw.drawline(mouses[i*160],c=colors[i%10])
        goal=goals[i*160]
        dw.drawgoal([goal[0],goal[1]+2000],c=colors[i%10])
    plt.show()

def drawline3d():
    ds=dataset.DataSet()
    ds.getTrainData()
    dw=datadraw.DataDraw('3d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    # x,y,t=mouses[0]
    colors=dw.getColorsValue()
    for i in range(10):
        dw.draw3dline(mouses[i*160],c=colors[i%10])
        goal=goals[i*160]
        dw.draw3dgoal([goal[0],goal[1]+2000,0],c=colors[i%10])
    plt.show()

def drawline3d2():
    ds=dataset.DataSet()
    ds.getTrainData()
    dw=datadraw.DataDraw('3d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    # x,y,t=mouses[0]
    colors=dw.getColorsValue()
    for i in range(10):
        dw.draw3dline(mouses[2600+i*40],c=colors[i%10])
        goal=goals[2600+i*40]
        dw.draw3dgoal([goal[0],goal[1]+2000,0],c=colors[i%10])
    plt.show()

def endgoalvector():
    ds=dataset.DataSet()
    ds.getTrainData()
    dw=datadraw.DataDraw('2d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    def getvecotr(mouse,goal):
        n=len(mouse[0])
        ex=mouse[0][n-1]
        ey=mouse[1][n-1]
        gx=goal[0]
        gy=goal[1]
        tmp=(gx-ex)**2+(gy-ey)**2
        tmp=tmp**0.5
        return [(gx-ex)/tmp,(gy-ey)/tmp]
    vector=[]
    for i in range(n):
        vector.append(getvecotr(mouses[i],goals[i]))
    vector=np.array(vector)
    dt=datadeal.DataTrain()
    # clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(2,2), random_state=1)
    clf = SVC()
    dt.trainTest(clf,vector,labels)
    # dw.drawbatchgoal(vector[:2600],c='b')
    # dw.drawbatchgoal(vector[2600:],c='r')
    # plt.show()
    # print vector

def axisy():
    ds=dataset.DataSet()
    ds.getTrainData()
    dw=datadraw.DataDraw('2d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]

    def getvecotr(mouse,goal):
        yn=len(mouse[1])
        y=mouse[1]
        if yn==1:
            return 0
        for i in range(yn)[-1:0:-1]:
            y[i]=y[i]-y[i-1]
        flag=1
        state=y[0]
        ychange=0
        for i in range(1,yn):
            if state*y[i]<0:
                ychange+=1
            state=y[i]
        return ychange

    vector=[]
    for i in range(n):
        vector.append(getvecotr(mouses[i],goals[i]))
    vector=np.array(vector,dtype=np.float)
    # dw.drawline([range(2600),vector[:2600]],c='b')
    # dw.drawline([range(2600,3000),vector[2600:]],c='r')
    # print vector[:2600].mean()
    # print vector[2600:].mean()

    # dw.drawbatchgoal(np.array([vector[:2600],labels[:2600]]).T,c='b')
    # dw.drawbatchgoal(np.array([vector[2600:],labels[2600:]]).T,c='r')
    # plt.show()

    # vector=np.array(vector)
    dt=datadeal.DataTrain()
    # # clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(2,2), random_state=1)
    clf = SVC()
    dt.trainTest(clf,vector.reshape([3000,1]),labels)

def axist():
    ds=dataset.DataSet()
    ds.getTrainData()
    dw=datadraw.DataDraw('2d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]

    def getvecotr(mouse,goal):
        tn=len(mouse[2])
        t=mouse[2]
        if tn==1:
            return 0
        return t[tn-1]-t[0]

    vector=[]
    for i in range(n):
        vector.append(getvecotr(mouses[i],goals[i]))
    vector=np.array(vector,dtype=np.float)
    # dw.drawline([range(2600),vector[:2600]],c='b')
    # dw.drawline([range(2600,3000),vector[2600:]],c='r')
    # print vector[:2600].mean()
    # print vector[2600:].mean()

    # dw.drawbatchgoal(np.array([vector[:2600],labels[:2600]]).T,c='b')
    # dw.drawbatchgoal(np.array([vector[2600:],labels[2600:]]).T,c='r')
    # plt.show()

    # vector=np.array(vector)
    dt=datadeal.DataTrain()
    # # clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(2,2), random_state=1)
    clf = SVC()
    dt.trainTest(clf,vector.reshape([3000,1]),labels)

def xspeed():
    ds=dataset.DataSet()
    ds.getTrainData()
    dw=datadraw.DataDraw('2d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]

    def getvecotr(mouse,goal):
        tn=len(mouse[2])
        t=mouse[2]
        x=mouse[0]
        if tn==1:
            return 0
        for i in range(tn)[-1:0:-1]:
            x[i]=x[i]-x[i-1]
            t[i]=t[i]-t[i-1]
            if t[i]>0:
                x[i]=x[i]/t[i]
            else:
                x[i]=0.0
        x=np.array(x)[1:]
        return x.mean()

    vector=[]
    for i in range(n):
        vector.append(getvecotr(mouses[i],goals[i]))
    vector=np.array(vector,dtype=np.float)
    # dw.drawline([range(2600),vector[:2600]],c='b')
    # dw.drawline([range(2600,3000),vector[2600:]],c='r')
    # print vector[:2600].mean()
    # print vector[2600:].mean()

    # dw.drawbatchgoal(np.array([vector[:2600],labels[:2600]]).T,c='b')
    # dw.drawbatchgoal(np.array([vector[2600:],labels[2600:]]).T,c='r')
    # plt.show()

    # vector=np.array(vector)
    dt=datadeal.DataTrain()
    # # # clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(2,2), random_state=1)
    clf = SVC()
    dt.trainTest(clf,vector.reshape([3000,1]),labels)

def startendvector():
    ds=dataset.DataSet()
    ds.getTrainData()
    dw=datadraw.DataDraw('2d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    def getvecotr(mouse,goal):
        n=len(mouse[0])
        ex=mouse[0][n-1]
        ey=mouse[1][n-1]
        et=mouse[2][n-1]
        bx=mouse[0][0]
        by=mouse[1][0]
        bt=mouse[2][0]
        if n>1:
            tmp=(bx-ex)**2+(by-ey)**2+(by-ey)**2
        else:
            tmp=(bx)**2+(by)**2+(by)**2
            bx=by=bt=0.0
        tmp=tmp**0.5
        if tmp<1e-3:
            tmp=1.0
        return [(ex-bx)/tmp,(ey-by)/tmp,(et-bt)/tmp]
    vector=[]
    for i in range(n):
        vector.append(getvecotr(mouses[i],goals[i]))
    vector=np.array(vector)
    dt=datadeal.DataTrain()
    # clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(2,2), random_state=1)
    clf = SVC()
    dt.trainTest(clf,vector,labels)
    # dw.drawbatchgoal(vector[:2600],c='b')
    # dw.drawbatchgoal(vector[2600:],c='r')
    # plt.show()
    # print vector

if __name__=="__main__":
    # drawline()
    # drawline3d2()
    # drawline3d()
    # endgoalvector()
    # axisy()
    # axist()
    # xspeed()
    startendvector()
    pass