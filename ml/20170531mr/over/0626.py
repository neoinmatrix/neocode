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

def getyn(mouse):
    yn=len(mouse[1])
    y=mouse[1]
    if yn==1:
        return [0.0,0.0]
        # return [0.0]
    for i in range(yn)[-1:0:-1]:
        y[i]=y[i]-y[i-1]
    flag=1
    state=y[0]
    ychange=0
    for i in range(1,yn):
        if state*y[i]<0:
            ychange+=1
        state=y[i]
    tmp=y/y.sum() 
    # return [float(tmp.std()*3.0)]
    return [float(ychange)/5.0,float(tmp.std()*3.0)]
def getlastt(mouse):
    xn=len(mouse[0])
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    idxget=0
    idxgetl=xn-1
    vs_all=0.0
    ve_all=0.0
    for i in range(4):
        if idxget<xn-1:
            idxget+=1
        if idxgetl>0:
            idxgetl-=1

        vs=(float(x[idxget])-float(x[idxget-1]))/1000
        ts=(float(t[idxget])-float(t[idxget-1]))/1000
        ts= ts if ts>1e-5 else 1.0
        vs/=ts
        idxb=idxgetl-1 if idxgetl>0 else 0
        ve=(float(x[idxgetl])-float(x[idxb]))/1000
        te=(float(t[idxgetl])-float(t[idxb]))/1000
        te= te if te>1e-5 else 1.0
        ve/=te

        vs_all+=vs
        ve_all+=ve


    percent=float(t[idxget])/float(t[xn-1])
    percent=percent*10.0 if percent*10.0<1.0 else 1.0
    
    percentl=float(t[idxgetl])/float(t[xn-1])
    percentl=percentl*1.0 if percentl*1.0<1.0 else 1.0

    return [percent/2,percentl/1.5,vs_all/5.0,ve_all/3]
def getlastangle(mouse):
    xn=len(mouse[0])
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    # idxget=0
    idxgetl=xn-1
    for i in range(5):
        # if idxget<xn-1:
        #     idxget+=1
        if idxgetl>0:
            idxgetl-=1

    cx=(float(x[xn-1])-float(x[idxgetl]))/1000
    cy=(float(y[xn-1])-float(y[idxgetl]))/1700/3
    ct=(float(t[xn-1])-float(t[idxgetl]))/1000
    if ct<1e-3:
        ct=1.0
    anglex=3.0*cx/ct
    angley=10.0*cy/ct
    # percentl=float(t[idxgetl])/float(t[xn-1])
    # percentl=percentl*1.0 if percentl*1.0<1.0 else 1.0
    return [anglex,angley]
def getlastgoal(mouse,goal):
    xn=len(mouse[0])
    x=mouse[0]
    y=mouse[1]
    tmpx=float(goal[0])-float(x[len(x)-1])
    tmpy=float(goal[1])-float(y[len(y)-1])
    return [tmpx/300,tmpy/3000]
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
    return [vx.max(),vx.min(),vx.mean(),vy.max(),vy.min(),vy.mean()]
    
def getvector(idx,mouse,goal,label):
    tmp=[]
    # mouse[1][0]/1700/3,
    xlen=len(mouse[0])
    tmp.extend([mouse[0][0]/mouse[0].max(),mouse[2][0]/mouse[2].max(),mouse[2][xlen-1]/mouse[2].max()])
    # ,mouse[2][0]/1000,mouse[2][xlen-1]/10000
    tmp.extend(getyn(mouse))
    tmp.extend(getlastt(mouse))
    tmp.extend(getlastangle(mouse))
    tmp.extend(getlastgoal(mouse,goal))
    tmp.extend(getmid(mouse))
    # tmp.extend(getspeed(mouse))
    return np.array(tmp).reshape([1,len(tmp)])

def test():
    ds=dataset.DataSet()
    ds.getTrainData()
    dw=datadraw.DataDraw('3d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    idxs=[1,100,1000,2000,2600,2700,2800,2800]
    for i in idxs:
        tmp=getspeed(mouses[i])
        tmpstr=""
        for v in tmp:
            tmpstr+="%5.2f "%v
        print tmpstr

    # x=vector[1]
    # xx=vector[1000]
    # y=vector[2700]
    # z=vector[2800]
    # for i in range(len(vector[0])):
    #     # print i,x[i],xx[i],y[i],z[i]
    #     if abs(x[i])!=0 and abs(abs(x[i])-abs(y[i]))/abs(x[i])>0.2:
    #         print i,x[i],y[i]
    # # print len(vector[0])
    exit()
    exit()

def assemble():
    ds=dataset.DataSet()
    ds.getTrainData()
    # dw=datadraw.DataDraw('3d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    for i in range(n):
        vector.append(getvector(1,mouses[i],goals[i],1)[0])
    vector=np.array(vector)

    dt=datadeal.DataTrain()
    clf = MLPClassifier(alpha=1e-8,activation='logistic', \
        hidden_layer_sizes=(16,16),random_state=0,solver='lbfgs',\
        max_iter=600)
    # clf = SVC(C=1.35,kernel='poly',degree=4,gamma=1,coef0=1.6)
    
    # False
    test=True
    if test==True:
        dt.trainTest(clf,vector,labels)
    else:
        dt.train(clf,vector,labels)
        dt.testResultAll(ds,getvector,savepath='./data/0626tmp.txt')


if __name__=="__main__":
    # print datadeal.calcScoreRerve(0.9472,19746.0)
    tmp=assemble()
    # test()
    pass