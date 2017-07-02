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
    

def getmidx(mouse,idxx):
    xn=len(mouse[0])
    # mid=xn/2
    # idxx=range(mid-2,mid+3)
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

    dt=dt if dt>1e-5 else 4200.0
    mt=mt if mt>1e-5 else 700.0
   

    a=dx/dt
    b=dy/dt
    c=dt/mt
    # a= a if abs(a)<1 else 0
    # # b= b if abs(b)<1 else 0
    # c= c if c<1 else 0
    return [a,b,c]

def getlastx(mouse):
    xn=len(mouse[0])
    mid=xn/2
    idxx=range(0,5)
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

    dt=dt if dt>1e-5 else 4200.0
    mt=mt if mt>1e-5 else 700.0
   

    a=dx/dt
    b=dy/dt
    c=dt/mt
    # a= a if abs(a)<1 else 0
    # b= b if abs(b)<1 else 0
    # c= c if c<1 else 0
    return [a,b,c]

def getrightup(mouse):
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
    # print twz
    # exit()
    return twz.reshape([1,9])


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

    # pass
    twz=twz/t[-1]
    # twz=twz**0.5
    # print twz
    # exit()
    return twz.reshape([1,6])


def getfeature(idx,mouse,goal,label,is_test=False,scal=''):
    tmp=[]
    t=mouse[2][0]
    # t = t if t <800 else 800.0                #  1. start t
    # t=t/800.0
    tmp.append(t)
    xlen=len(mouse[0])
    # # size=float(xlen)/300.0
    # # size=size**0.5
    tmp.append(xlen) # size of mouse          # 2. size of mouse

    startx=float(mouse[0][0])
    # startx=170 if startx<170 else startx
    # startx=1840 if startx>1840 else startx
    # startx=(startx-170)/1670
    # startx=startx**0.2
    # startx=0 if startx<1e-3 else startx
    tmp.append(startx) # start x             # 3. start x

    starty=float(mouse[1][0])
    # starty=1898 if starty<1898 else starty
    # starty=3035 if starty>3035 else starty
    # starty=(starty-1898)/1137
    # starty=0 if starty<1e-3 else starty
    tmp.append(starty) # start x              # 4.start y

    tstd=mouse[2].std()                        # 5. t std
    # tstd=0 if tstd<0 else tstd
    # tstd=17132 if tstd>17132 else tstd
    # tstd=tstd/17132.0
    # tstd=tstd**0.3
    # tstd=0 if tstd<1e-3 else tstd
    tmp.append(tstd) # start x

    tuse=mouse[2][-1]
    # tuse=tuse/37132.0
    # tuse=tstd**0.3
    # tuse=0 if tuse<1e-5 else tuse             # 6. last t      
    tmp.append(tuse) # start x

    ex=mouse[0][-1]
    ey=mouse[1][-1]

    gx=goal[0]
    gy=goal[1]

    dx=(ex-gx)
    dy=(ey-gy)                      # 7.8 9  last mouse with goal positions
    tmp.append(dx)
    tmp.append(dy)    
    tmp.append(-1*dx)
    tmp.append(-1*dy)    
    dis=dx**2+dy**2
    dis=dis**0.5                          
    tmp.append(dis)                              

    xn=len(mouse[0])
    mid=xn/2
    idxx=range(mid-2,mid+3)
    x,y,t=getmidx(mouse,idxx)                 
    tmp.append(x)
    tmp.append(y)
    tmp.append(t)  

    idxx=range(0,5)
    x,y,t=getmidx(mouse,idxx)                 
    tmp.append(x)
    tmp.append(y)
    tmp.append(t)  

    idxx=range(xn-6,xn-1)
    x,y,t=getmidx(mouse,idxx)                 
    tmp.append(x)
    tmp.append(y)
    tmp.append(t)  


    twz=getrightup(mouse)                   # 12,13 mid five points x,y speed 
    # print twz[0].tolist()
    # exit()
    # tmp.append(twz[0].tolist())
    tmp.extend(twz[0].tolist())    

    twz=getplr(mouse)     
    tmp.extend(twz[0].tolist())
 

    # yu,yd,yuu,ydd=getrightup(mouse)                   # 12,13 mid five points x,y speed 
    # tmp.append(yu)
    # tmp.append(yd)
    # tmp.append(yuu)
    # tmp.append(ydd)
    # tmp.append(t)  
    # print tmp 
    # tmp[2]=tmp[2]*10
    # print tmp 
    # exit()

    if is_test==True:
        tmp=scal.transform([tmp])
        tmp=tmp[0].tolist()
        # print tmp
        # exit()
        # is_test

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
    np.set_printoptions(formatter={'float':lambda x: "%5.2f"%float(x)})
    scaler_vector=vector
    vector = preprocessing.scale(vector)

    # print vector[0:10]
    # print vector[:,0].min()
    # print vector[:,0].max()
    # print vector[:,0].mean()
    # print vector[:,0].std()
    # exit()
    # print vector[1000:1010]
    # print "=============="
    # # print vector[2700:2710]
    # print vector[2800:2810]
    # exit()
    # scaler = preprocessing.StandardScaler().fit(X)
    # for i in range(3):
    #     a=vector[0:2600,i].mean()
    #     b=vector[2600:3000,i].mean()

    #     ar=vector[0:2600:,i].std()
    #     br=vector[2600:3000:,i].std()
    #     print a-ar," ",a+ar
    #     print b-br," ",b+br

    #     # print vector[0:2600,i].mean()
    #     # print vector[2600:3000,i].mean()
    #     # print vector[0:2600:,i].std()
    #     # print vector[2600:3000:,i].std()
    #     print i,"============"
    # exit()
 
    dt=datadeal.DataTrain()
    # clf = MLPClassifier(alpha=1e-8,
    #     activation='logistic', \
    #     hidden_layer_sizes=(33,3),random_state=1,solver='lbfgs',\
    #     max_iter=1200)
    clf = MLPClassifier(alpha=1e-3,
        activation='logistic', \
        hidden_layer_sizes=(10,16),random_state=0,solver='lbfgs',\
        max_iter=1000)
    # lbfgs ,kernel='poly',degree=4,gamma=1,coef0=1.6
    # clf = SVC(C=20.35)
    
    # False
    test=False
    if test==True:
        dt.trainTest(clf,vector,labels,10.0)
    else:
        scaler = preprocessing.StandardScaler().fit(scaler_vector)
        dt.train(clf,vector,labels)
        dt.testResultAll(ds,getfeature,savepath='./data/0629tmp.txt',stop=1200,scal=scaler)


if __name__=="__main__":
    # print datadeal.calcScoreRerve(0.9472,19746.0)
    # print "sdfsdf"
    # exit(0)
    tmp=assemble()
    # test()
    pass