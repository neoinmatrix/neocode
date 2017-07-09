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

def getMMMS(d):
    return [d.min(),d.max(),d.mean(),d.std()]

def getfivex(d,idxx):
    xn=len(d)
    for i in range(5):
        if idxx[i]<0:
            idxx[i]=0
        if idxx[i]>(xn-1):
            idxx[i]=xn-1
    d=np.array(d)
    return d[idxx].tolist()

def getangle(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    xn=len(mouse[0])
    vx=[0.0]
    vy=[0.0]
    angle_arr=[0.0]
    aspeed_arr=[0.0]
    for i in range(1,xn):
        if i+1>=xn:
            break
        else:
            vx1=x[i+1]-x[i]
            vy1=y[i+1]-y[i]
            vx2=x[i]-x[i-1]
            vy2=y[i]-y[i-1]
            dt=t[i+1]-t[i-1]
            angle=(vx1*vx2+vy1*vy2)
            if vx1==0 and vy1==0:
                continue
            if vx2==0 and vy2==0:
                continue
            if dt==0:
                continue
            angle/=(vx1**2+vy1**2)**0.5
            angle/=(vx2**2+vy2**2)**0.5
            speed=angle/dt
        angle_arr.append(angle)
        aspeed_arr.append(speed)
    angle_arr=np.array(angle_arr)
    aspeed_arr=np.array(aspeed_arr)
    result=[]
    result.extend(getMMMS(angle_arr))
    result.extend(getMMMS(aspeed_arr))
    
    idxx=range(xn-5,xn)
    tmp=np.array(getfivex(angle_arr,idxx))
    result.extend([tmp.mean()])  
    tmp=np.array(getfivex(aspeed_arr,idxx))
    result.extend([tmp.mean()])
    return result


def getfive(mouse,idxx):
    xn=len(mouse[0])
    for i in range(5):
        if idxx[i]<0:
             idxx[i]=0
        if idxx[i]>(xn-1):
            idxx[i]=xn-1

    dt=mouse[2][idxx[-1]]-mouse[2][idxx[0]]
    dx=mouse[0][idxx[-1]]-mouse[0][idxx[0]]
    dy=mouse[1][idxx[-1]]-mouse[1][idxx[0]]
    mt=mouse[2][-1]

    dt=dt if dt>1e-5 else 4200.0
    mt=mt if mt>1e-5 else 700.0
    a=dx/dt
    b=dy/dt
    c=dt/mt
    return [a,b,c]

def get_derivative(mouse):
    xn=len(mouse[0])
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    vxs=[0.0]
    vys=[0.0]
    for i in range(1,xn-1):
        dt=t[i]-t[i-1]
        dx=x[i]-x[i-1]
        dy=y[i]-y[i-1]
        if dt==0:
            dt=1.0
        vx=dx/dt
        vy=dy/dt
        vxs.append(vx)
        vys.append(vy)
    axs=[0.0]
    ays=[0.0]
    for i in range(1,xn-2):
        ddx=vxs[i]-vxs[i-1]
        ddy=vys[i]-vys[i-1]
        dt=t[i+1]-t[i-1]
        if dt==0:
            dt=10000.0
        vvx=ddx/dt
        vvy=ddy/dt
        axs.append(vvx)
        ays.append(vvy)
    # idxx=range(xn-5,xn)
    # idxx=range(0,5)
    xnt=int((xn-1)/2)
    idxx=[i for i in range(xnt-2,xnt+3)]
    # print idxx
    # print xn
    lax=getfivex(axs,idxx)
    lay=getfivex(ays,idxx)
    lax=np.array(lax)
    lay=np.array(lay)
    # avr_ax=float(sum(lax))/float(len(lax))
    # avr_ay=float(sum(lay))/float(len(lay))
    return [lax.min(),lax.std(),lay.min(),lay.std()]

def get_mv(mouse):
    xn=len(mouse[0])
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    mvxs=[0.0]
    for i in range(1,xn-1):
        dt=t[i]-t[i-1]
        dx=x[i]-x[i-1]
        dy=y[i]-y[i-1]
        if dt==0:
            dt=100000.0
        mvx=dy*dx/dt
        mvx=mvx if mvx<600.0 else 600.0
        mvx=mvx if mvx>-600.0 else -600.0
        mvxs.append(mvx)
    mvxs=np.array(mvxs)
    return [mvxs.min(),mvxs.max(),mvxs.mean(),mvxs.std()]

def get_entropy(mouse):
    xn=len(mouse[0])
    # x=mouse[0]
    # y=mouse[1]
    t=mouse[2]
    ta=t[-1]
    entropy=0.0
    for i in range(1,xn-1):
        dt=t[i]-t[i-1]
        if dt==0:
            dt=1.0
        if dt<0:
            entropy=0.0
            continue
        p=float(dt)/float(ta)
        entropy+=np.log(p)*p
    entropy=entropy if entropy<0 else 0.0
    return [entropy]


def analyst_xyt2():
    def getxyt(idx,mouse,goal,label):
        tmp=[]
        angles=get_entropy(mouse)
        tmp.extend(angles)
        return np.array(tmp).reshape([1,len(tmp)])
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    for i in range(n):
        vector.append(getxyt(1,mouses[i],goals[i],1)[0])
    vector=np.array(vector)

    def draw(j):
        import matplotlib.pyplot as plt
        size=[]
        for i in range(n):
            size.append(vector[i,j])
        size=np.array(size)    
        plt.plot(range(3000),size)
        plt.show()
        # plt.title(name[j])
        # plt.savefig('./data/xyt/%s.png'%name[j])
        # plt.clf()
    draw(0)

if __name__=="__main__":
    # analyst_xyt()
    analyst_xyt2()
    # print np.log(1.0/4.0)*(1.0/4.0)