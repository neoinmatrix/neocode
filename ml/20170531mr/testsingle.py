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

def getmid(mouse):
    xn=len(mouse[0])
    mid=xn/2
    idxx=range(mid-2,mid+3)
    for i in range(5):
        if idxx[i]<0:
             idxx[i]=0
        if idxx[i]>(xn-1):
            idxx[i]=xn-1

    # print idx
    dt=mouse[2][idxx[-1]]-mouse[2][idxx[0]]
    dx=mouse[0][idxx[-1]]-mouse[0][idxx[0]]
    dy=mouse[1][idxx[-1]]-mouse[1][idxx[0]]
    mt=mouse[2][-1]

    dt=dt if dt>1e-5 else 1.0
    mt=mt if mt>1e-5 else 1.0
   
    a=dx/dt
    b=dy/dt
    c=dt/mt
    return [a,b,c]

def getspeed(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    xn=len(mouse[0])
    vx=[0.0]
    vy=[0.0]
    for i in range(1,xn):
        vspeed=(float(x[i])-float(x[i-1]))
        vyspeed=(float(y[i])-float(y[i-1]))
        dt=float(t[i])-float(t[i-1])
        if dt==0:
            continue
        vspeed/=dt
        vyspeed/=dt
        vx.append(vspeed)
        vy.append(vyspeed)
    vx=np.array(vx)
    vy=np.array(vy)
    minvx=vx.min()
    minvx=minvx if minvx>-1 else -1 
    minvx*=-1
    return [vx.max(),minvx,vx.mean()/10.0,vy.max(),vy.min(),vy.mean()]
 

def getfeature(idx,mouse,goal,label):
    tmp=[]
    x=getspeed(mouse)
    # print x
    # exit(0)
    tmp.append(x[1])
    # for v in x:
    #     tmp.append(v)

    # a,b,c=getmid(mouse)
    # tmp.append(a)
    # tmp.append(b)
    # tmp.append(c)
    # ex=mouse[0][-1]
    # ey=mouse[1][-1]

    # gx=goal[0]
    # gy=goal[1]
    # # distance=(ex-gx)**2+(ey-gy)**2
    # # distance=distance**0.5
    # tmp.append(ex-gx)
    # tmp.append(ey-gy)
    return np.array(tmp).reshape([1,len(tmp)])

def assemble():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    for i in range(n):
        vector.append(getfeature(1,mouses[i],goals[i],1)[0])
    vector=np.array(vector)
    # print vector[0:10]
    print vector[0:50]
    print vector[2700:2710]
    print vector[2800:2830]
    # exit()
 
    dt=datadeal.DataTrain()
    count=0
    for i in vector:
        if i==0:
            count+=1
    print count
    exit(0)
    # import sklearn.logistic as logistic
    # clf=SVC()
    # clf = MLPClassifier(alpha=1e-4,activation='logistic', \
    #     hidden_layer_sizes=(16,16),random_state=0,solver='lbfgs',\
    #     max_iter=600)
    # clf = SVC(C=1.35,kernel='poly',degree=4,gamma=1,coef0=1.6)
    
    # False
    test=True
    if test==True:
        dt.trainTest(clf,vector,labels)
    else:
        dt.train(clf,vector,labels)
        dt.testResultAll(ds,getfeature,savepath='./data/0629tmp.txt')


if __name__=="__main__":
    # print datadeal.calcScoreRerve(0.9472,19746.0)
    tmp=assemble()
    # test()
    pass