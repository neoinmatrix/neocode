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
from sklearn import preprocessing
from sklearn.decomposition import PCA

#========common=================

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

#===============================

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

def gettoward(mouse):
    n=len(mouse[0])
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]

    twz=np.zeros([3,3],dtype='float')
    for i in range(1,n-1):
        dt=t[i]-t[i-1]
        if x[i]-x[i-1]==0:
            if  y[i]-y[i-1]==0:
                twz[0,0]+=dt
            elif y[i]-y[i-1]>0:
                twz[0,1]+=dt
            else:
                twz[0,2]+=dt
        elif x[i]-x[i-1]>0:
            if  y[i]-y[i-1]==0:
                twz[1,0]+=dt
            elif y[i]-y[i-1]>0:
                twz[1,1]+=dt
            else:
                twz[1,2]+=dt
        else:
            if  y[i]-y[i-1]==0:
                twz[2,0]+=dt
            elif y[i]-y[i-1]>0:
                twz[2,1]+=dt
            else:
                twz[2,2]+=dt

    # pass
    twz=twz/float(t[-1])
    # twz=twz**0.5
    return twz.reshape([1,9])[0]

def getplr(mouse):
    n=len(mouse[0])
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]

    twz=np.zeros([2,3],dtype='float')
    for i in range(1,n-1):
        dt=t[i]-t[i-1]
        if x[i]-x[i-1]==0:
            twz[0,0]+=dt
        elif x[i]-x[i-1]>0:
            twz[0,1]+=dt
        else:
            twz[0,2]+=dt

        if  y[i]-y[i-1]==0:
            twz[1,0]+=dt
        elif y[i]-y[i-1]>0:
            twz[1,1]+=dt
        else:
            twz[1,2]+=dt

    twz=twz/t[-1]
    # twz=twz**0.5
    return twz.reshape([1,6])[0]

def getStatistic(mouse):
    x=mouse[0]
    y=mouse[0]
    t=mouse[0]
    def getanalyst(tmp):
        return [tmp.min(),tmp.max(),tmp.mean(),tmp.std()]
    analyst=[]
    analyst.extend(getanalyst(x))
    analyst.extend(getanalyst(y))
    analyst.extend(getanalyst(t))
    return analyst

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


def getfeature(idx,mouse,goal,label):
    tmp=[]
    tmp.append(float(mouse[0][0]))                     # 2. start x

    tmp.append(len(mouse[0]))                          # 1. size of mouse

    
    tmp.append(float(mouse[1][0]))                      # 3. start y

    tmp.append(mouse[2][0])                            # 4. start t
    tmp.append(mouse[2].std())                         # 5. start t.std
    tmp.append(mouse[2][-1])                           # 6. last t 

    ex=mouse[0][-1]
    ey=mouse[1][-1]
    gx=goal[0]
    gy=goal[1]
    dx=(ex-gx)
    dy=(ey-gy) 
    
    #dx dy * -2   +distance                           # 7.8 9  last mouse with goal positions
    tmp.append(dx)
    tmp.append(dy)    
    tmp.append(-1*dx)
    tmp.append(-1*dy)  
    dis=dx**2+dy**2
    dis=dis**0.5                          
    tmp.append(dis)                              

    # 9 + 9 + 6 = 24
    xn=len(mouse[0])
    mid=int(xn/2)                #  x y t  in mid begin end position 3*3
    idxx=range(mid-2,mid+3)
    xyt=getfive(mouse,idxx)                 
    tmp.extend(xyt) 

    idxx=range(0,5)
    xyt=getfive(mouse,idxx)                 
    tmp.extend(xyt) 

    idxx=range(xn-6,xn-1)
    xyt=getfive(mouse,idxx)                 
    tmp.extend(xyt)  

    twz=gettoward(mouse)                   # 12,13 mid five points x,y speed 
    tmp.extend(twz.tolist())    

    twz=getplr(mouse)     
    tmp.extend(twz.tolist())

    statistic=getStatistic(mouse)
    tmp.extend(statistic)

    angles=getangle(mouse)
    tmp.extend(angles)

    ddxy=get_derivative(mouse)
    tmp.extend(ddxy)
  
    mv=get_mv(mouse)
    tmp.extend(mv)

    entropy=get_entropy(mouse)
    tmp.extend(entropy)

    return np.array(tmp).reshape([1,len(tmp)])


def printsomething(vector):
    np.set_printoptions(formatter={'float':lambda x: "%5.2f"%float(x)})
    # print vector[0:10]
    # print vector[:,0].min()
    # print vector[:,0].max()
    # print vector[:,0].mean()
    # print vector[:,0].std()
    # exit()
    print vector[1000:1010]
    print "=============="
    # print vector[2700:2710]
    print vector[2800:2810]
    exit()
    scaler = preprocessing.StandardScaler().fit(X)
    for i in range(3):
        a=vector[0:2600,i].mean()
        b=vector[2600:3000,i].mean()

        ar=vector[0:2600:,i].std()
        br=vector[2600:3000:,i].std()
        print a-ar," ",a+ar
        print b-br," ",b+br

        # print vector[0:2600,i].mean()
        # print vector[2600:3000,i].mean()
        # print vector[0:2600:,i].std()
        # print vector[2600:3000:,i].std()
        print i,"============"
    exit()

def get_X_PN_isMachine(mouse):
    x=mouse[0]
    n=len(x)
    for i in range(1,n):
        if x[i-1]>x[i]:
            return False
    return True


def main():
    ds=dataset.DataSet(testfp='./data/dsjtzs_txfz_testB.txt')
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    for i in range(n):
        mouse=mouses[i]
        if get_X_PN_isMachine(mouse)==False:
            continue
        print i," ",mouse[1][0],mouse[1][-1]-goals[i][1]

if __name__=="__main__":
    # def cal(a,b):
    #     return float(a)/float(a+b)
    main()
    # print(cal(386,24))
    # print(cal(386,15))
    # print(cal(382,29))
    # import time
    # start =time.clock()
    # main()
    # end = time.clock()
    # print('Running time: %s Seconds'%(end-start))

    # print datadeal.calcScoreRerve(0.9480,20045)
    pass