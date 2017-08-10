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

def get_angle_sharp(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])
    # x=x/x.max()
    # y=y/y.max()
    # t=t/t.max()

    angle_dist=np.zeros(3,dtype=np.float)
    sharp=np.array([0])
    for i in range(1,n):
        if i+1>=n:
            break
        else:
            vx1=x[i+1]-x[i]
            vy1=y[i+1]-y[i]
            # vt1=t[i+1]-t[i]
            vx2=x[i]-x[i-1]
            vy2=y[i]-y[i-1]
            # vt2=t[i]-t[i-1]
            dt=t[i+1]-t[i-1]
            angle=vx1*vx2+vy1*vy2

            # angle3d=angle+vt1*vt2
            if dt==0:
                continue
            r1=(vx1**2+vy1**2)**0.5
            r2=(vx2**2+vy2**2)**0.5

            if r1==0 or r2==0:
                continue
            angle/=r1
            angle/=r2
            # print angle,dt,r1,r2
            if angle>-1.1 and angle<=-0.3:
                angle_dist[0]+=1
            elif angle>-0.3 and angle<=0.3:
                angle_dist[1]+=1
            elif angle>0.3:
                angle_dist[2]+=1
            if angle<0:
                rr=r1 if r1<r2 else r2
                sharp=np.append(sharp,rr)
    
    n=float(sum(angle_dist))
    feat=np.array([])
    feat=np.append(feat,angle_dist)
    feat=np.append(feat,float(len(sharp)))
    feat=np.append(feat,sharp.mean())
    feat=np.append(feat,sharp.max())
    # print sharp
    return feat.flatten()

def get_feats(idx,mouse,goal,label):
    feats=np.array([])
    feats=np.append(feats,get_angle_sharp(mouse))
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
            if i in range(2600,2900):
                continue
            idxs.append(i)
            vtr_sharp.append(get_feats(1,mouses[i],goals[i],1))
            lb_sharp.append(labels[i])
    vtr_sharp=np.array(vtr_sharp)    
    lb_sharp=np.array(lb_sharp)    
    # print vtr_sharp.shape

    # print vtr_sharp[1]
    # print vtr_sharp[1000]
    # print idxs[1000]
    # plt.plot(range(6),vtr_sharp[1])
    # plt.plot(range(6),vtr_sharp[1000])
    # plt.show()
    # exit()
    # mouse=mouses[idxs[1]]
    # plt.plot(mouse[0],mouse[1],c='r')
    # mouse=mouses[idxs[1000]]
    # plt.plot(mouse[0],mouse[1],c='g')
    # plt.show()

    vtr_sharp = preprocessing.scale(vtr_sharp)
    # plt.title('the sharp shape of machine')
    # for i in range(len(vtr_sharp)):
    #     if lb_sharp[i]==0:
    #         plt.plot(range(6),vtr_sharp[i],c='r')
    #     else:
    #         plt.plot(range(6),vtr_sharp[i],c='g')
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