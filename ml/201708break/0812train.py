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

def flt_mac(mouse):
    x=mouse[0]
    n=len(x)
    for i in range(1,n):
        if x[i-1]>x[i]:
            return False
    return True

def get_nd(d,idxx):
    n=len(d)
    for i in range(len(idxx)):
        if idxx[i]<0:
            idxx[i]=0
        if idxx[i]>(n-1):
            idxx[i]=n-1
    d=np.array(d)
    return d[idxx].flatten()

def get_actions(mouse):
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

def get_towards(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])

    tmpx=x[0]
    tmpy=y[0]
    xy=[]
    for i in range(1,n):
        if tmpx==x[i] or tmpy==y[i]:
            continue
        xy.append([x[i],y[i]])
        tmpx=x[i]
        tmpy=y[i]
    tan_arr=np.array([0])
    for i in range(1,len(xy)):
        sx,sy=xy[i-1]
        ex,ey=xy[i]
        tan=(ey-sy)/(ex-sx)
        tan_arr=np.append(tan_arr,tan)

    tmptan=tan_arr[0]
    change=np.array([0,0,0])
    for i in range(1,len(tan_arr)):
        if tan_arr[i]>0:
            change[0]+=1
        else:
            change[1]+=1
        if tan_arr[i]*tmptan<0:
            change[2]+=1
        tmptan=tan_arr[i]
    feat=[]
    feat=np.array([])
    if n>1:
        startdt=(t[1]-t[0])/t[-1]
    else:
        startdt=0
    # feat=np.append(feat,startdt)
    feat=np.append(feat,change/float(n))
    feat=np.append(feat,float(n))
    feat=np.append(feat,len(xy))

    return feat.flatten()

def get_cxcy(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])
    # x=x/x.max()
    # y=y/y.max()
    # t=t/t.max()
    tmpx=x[0]
    tmpy=y[0]
    tmpt=0
    countx=0.0
    county=0.0
    tarr=np.array([0])
    for i in range(1,n):
        if tmpx!=x[i]:
            tmpx=x[i]
            countx+=1
        if tmpy!=y[i]:
            tmpy=y[i]
            county+=1
            tarr=np.append(tarr,tmpt)
            tmpt=0
        tmpt+=(t[i]-t[i-1])
    if n>1:
        startdt=(t[1]-t[0])/t[-1]
    else:
        startdt=0

    feat=np.array([])
    feat=np.append(feat,[startdt,countx/float(n),county/float(n),n])
    feat=np.append(feat,[tarr.max(),tarr.mean(),tarr.std()])
    return feat.flatten()

def get_all(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])
    x=x/max(x)
    y=y/max(y)
    t=t/max(t)
    
    vxt=[0.0]
    for i in range(0,n):
        if i+1>=n:
            break
        else:
            vx=x[i+1]-x[i]
            dt=t[i+1]-t[i]
            if dt==0:
                continue
            angle=vx/dt
            vxt.append(angle)
    feat=[]

    idx=range(len(vxt)-5,len(vxt))
    lastfive=get_nd(vxt,idx)
    feat.extend(lastfive)
    lastfive=np.array(lastfive)
    feat.extend([lastfive.std(),lastfive.mean()])

    # idx=range(0,5)
    # beginfive=getfivex(vxt,idx)
    # beginfive=np.array(beginfive)
    # feat.extend([beginfive.std(),beginfive.mean()])

    trate=sum(t[:5])/abs(t[-1])
    feat.append(trate)
 
    idx=range(0,3)
    t3=get_nd(t,idx)
    feat.extend(t3)

    if  len(mouse[2])>1:
        feat.append(1.0 if mouse[2][1]>500 else 0.0)
    else:
        feat.append(0.0)

    # first five angle is concave convex
    xfive=get_nd(x,range(0,5))
    tfive=get_nd(t,range(0,5))
    ax=xfive[4]-xfive[0]
    at=tfive[4]-tfive[0]
    mx=(xfive[4]+xfive[0])/2
    mt=(tfive[4]+tfive[0])/2
    bx=mx-xfive[2]
    bt=mt-tfive[2]
    angle=ax*bx+at*bt
    feat.append(angle)

    return np.array(feat).flatten()

def get_sharp(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])

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
    
    feat=np.array([])
    feat=np.append(feat,angle_dist)
    feat=np.append(feat,float(len(sharp)))
    feat=np.append(feat,sharp.mean())
    feat=np.append(feat,sharp.max())
    return feat.flatten()

def get_diff(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])
    def statistic(tmp):
        return [len(tmp),tmp.min(),tmp.max(),tmp.mean(),tmp.std()]
    dx=np.array([0])
    dy=np.array([0])
    dt=np.array([0])
    for i in range(1,n):
        dx=np.append(dx,x[i]-x[i-1])
        dy=np.append(dy,y[i]-y[i-1])
        dt=np.append(dt,t[i]-t[i-1])
    feat=np.array([])
    feat=np.append(feat,statistic(dx))
    feat=np.append(feat,statistic(dy))
    feat=np.append(feat,statistic(dt))
    return feat.flatten()

def get_yept(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])
    ynum={}
    for i in range(0,n-1):
        ynum.setdefault(y[i],0.0)
        ynum[y[i]]+=(t[i+1]-t[i])
    ept=0.0
    for k in ynum:
        p=float(ynum[k])/float(t[-1])
        if p==0:
            continue
        ept+=p*np.log(p)
    return np.array([ept,n,len(ynum)]).flatten()


def get_feats(idx,mouse,goal,label):
    feats=np.array([])
    feats=np.append(feats,get_yept(mouse))
    # feats=np.append(feats,get_towards(mouse))
    # feats=np.append(feats,get_actions(mouse))
    # feats=np.append(feats,get_cxcy(mouse))
    # feats=np.append(feats,get_all(mouse))
    # feats=np.append(feats,get_diff(mouse))

    # feats=np.append(feats,get_sharp(mouse))
    return feats.flatten()
    # return np.array(tmp).reshape([1,len(tmp)])

def main():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    # print get_yept(mouses[2601])
    # exit()
    # print get_sharp(mouses[2960])
    vtr_sharp=[]
    lb_sharp=[]
    idxs=[]
    for i in range(n):
        if flt_mac(mouses[i])==True:
            if i not in range(2600,2800) and i not in range(2600):
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