# coding=utf-8
import dataset
import datadeal
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import  metrics
import matplotlib.pyplot as plt 
from sklearn import preprocessing

def get_xpn_mac(mouse):
    x=mouse[0]
    n=len(x)
    for i in range(1,n):
        if x[i-1]>x[i]:
            return False
    return True

def filter_xnp(data):
    pass
    return get_xpn_mac(data[1])

def get_nd(d,idxx):
    n=len(d)
    for i in range(len(idxx)):
        if idxx[i]<0:
            idxx[i]=0
        if idxx[i]>(n-1):
            idxx[i]=n-1
    d=np.array(d)
    return d[idxx].flatten()

def get_action_shape(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])

    tmpy=y[0]
    county=0.0
    vxt=np.array([0])
    for i in range(1,n):
        vx=x[i]-x[i-1]
        dt=t[i]-t[i-1]
        if dt==0:
            continue
        angle=vx/dt
        vxt=np.append(vxt,angle)
        if tmpy!=y[i]:
            county+=1
            tmpy=y[i]
    idx=range(0,5)
    start=get_nd(vxt,idx)
    idx=range(len(vxt)-5,len(vxt))
    over=get_nd(vxt,idx)

    feat=[]
    feat=np.array([])
    if n>1:
        startdt=(t[1]-t[0])/t[-1]
    else:
        startdt=0
    feat=np.append(feat,startdt)
    feat=np.append(feat,county/float(n))
    feat=np.append(feat,[start.mean(),start.max()])
    feat=np.append(feat,[over.mean(),over.max()])

    return feat.flatten()

def get_feats(idx,mouse,goal,label):
    feats=np.array([])
    # feats=np.append(feats,get_angle_sharp(mouse))
    feats=np.append(feats,get_action_shape(mouse))
    return feats.flatten()
    # return np.array(tmp).reshape([1,len(tmp)])

def main():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]

    # print get_angle_sharp(mouses[2960])
    vtr_sharp=[]
    lb_sharp=[]
    idxs=[]
    for i in range(n):
        if get_xpn_mac(mouses[i])==True:
            if i not in range(2600,2700) and i not in range(2600):
                continue
            idxs.append(i)
            vtr_sharp.append(get_feats(1,mouses[i],goals[i],1))
            lb_sharp.append(labels[i])
    vtr_sharp=np.array(vtr_sharp)    
    lb_sharp=np.array(lb_sharp)    
    # print vtr_sharp.shape
    # exit(0)
    vtr_sharp = preprocessing.scale(vtr_sharp)
    # print vtr_sharp[1]
    # print vtr_sharp[1000]
    # print idxs[1000]
    # plt.plot(range(3),vtr_sharp[1])
    # plt.plot(range(3),vtr_sharp[1000])
    # plt.show()
    # exit()
    # mouse=mouses[idxs[1]]
    # plt.plot(mouse[0],mouse[1],c='r')
    # mouse=mouses[idxs[1000]]
    # plt.plot(mouse[0],mouse[1],c='g')
    # plt.show()
    # exit()
    
    # plt.title('the cx cy machine')
    # for i in range(len(vtr_sharp)):
    #     if lb_sharp[i]==0:
    #         plt.plot(range(3),vtr_sharp[i],c='r')
    #     else:
    #         plt.plot(range(3),vtr_sharp[i],c='g')
    # plt.show()
    # exit()
    # print vtr_sharp[0]
    # print vtr_sharp[-1]
    dt=datadeal.DataTrain()
    # about 17 w
    clf=SVC(C=1.5)
    dt.trainTest(clf,vtr_sharp,lb_sharp,10.0)
       
if __name__=="__main__":
    main()
    pass