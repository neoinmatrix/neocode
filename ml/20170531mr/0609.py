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

def getvector(idx,mouse,goal,label):
    def goalend(mouse,goal):
        n=len(mouse[0])
        ex=mouse[0][n-1]
        ey=mouse[1][n-1]
        gx=goal[0]
        gy=goal[1]
        tmp=(gx-ex)**2+(gy-ey)**2
        tmp=tmp**0.5
        return [(gx-ex)/tmp,(gy-ey)/tmp]
    def startend(mouse):
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
    def getyn(mouse):
        yn=len(mouse[1])
        y=mouse[1]
        if yn==1:
            return [0]
        for i in range(yn)[-1:0:-1]:
            y[i]=y[i]-y[i-1]
        flag=1
        state=y[0]
        ychange=0
        for i in range(1,yn):
            if state*y[i]<0:
                ychange+=1
            state=y[i]
        return [float(ychange)]
    def gett(mouse):
        tn=len(mouse[2])
        t=mouse[2]
        if tn==1:
            return [0.0]
        return [(t[tn-1]-t[0])/1000]
    def getxspeed(mouse):
        tn=len(mouse[2])
        t=mouse[2]
        x=mouse[0]
        if tn==1:
            return [0]
        for i in range(tn)[-1:0:-1]:
            x[i]=x[i]-x[i-1]
            t[i]=t[i]-t[i-1]
            if t[i]>0:
                x[i]=x[i]/t[i]
            else:
                x[i]=0.0
        x=np.array(x)[1:]
        return [x.mean()/10]
    def getendstart(mouse,goal):
        n=len(mouse[0])
        # if n<=1:
        #     pass
            # return 
        bx=mouse[0][0]
        by=mouse[1][0]
        ex=mouse[0][n-1]
        ey=mouse[1][n-1]
        gx=goal[0]
        gy=goal[1]
        tmpx=bx-(ex-gx)
        tmpy=by-(ey-gy)
        return [tmpx,tmpy]
    tmp=[]
    # tmp.extend([mouse[0][0]/1000,mouse[1][0]/1000])
    tmp.extend(getendstart(mouse,goal))
    # tmp.extend(startend(mouse))
    # tmp.extend(getyn(mouse))
    # tmp.extend(gett(mouse))
    # tmp.extend(getxspeed(mouse))
    # np.array([xarr[0],yarr[0]]).reshape([1,2])
    return np.array(tmp).reshape([1,len(tmp)])
   
def assemble():
    ds=dataset.DataSet()
    ds.getTrainData()
    dw=datadraw.DataDraw('2d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    # print mouses[492],goals[492]
    for i in range(n):
        vector.append(getvector(1,mouses[i],goals[i],1)[0])
    vector=np.array(vector)
    # print vector.shape
    # dw =datadraw.DataDraw('2d')
    # dw.drawbatchgoal(vector[0:2600],c='b')
    # dw.drawbatchgoal(vector[2600:],c='r')
    # plt.show()
    dt=datadeal.DataTrain()
    # # clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(40), random_state=1)
    clf = SVC(C=0.65)
    dt.trainTest(clf,vector,labels)

if __name__=="__main__":
    assemble()

    pass