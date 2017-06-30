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
    
def analyst_xyt():
    def getxyt(idx,mouse,goal,label):
        tmp=[]
        xlen=len(mouse[0])
        tmp.append(xlen) # len 
        for i in range(3):
            tmp.append(mouse[i].min()) # x min
            tmp.append(mouse[i].max()) # x max
            tmp.append(mouse[i].mean()) # x max
            tmp.append(mouse[i].std()) # x max
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

    name=["mouse_size",\
    "x_min","x_max","x_mean","x_std",\
    "y_min","y_max","y_mean","y_std",\
    "t_min","t_max","t_mean","t_std",\
    ]
    allx='name\tmin\tmax\tmean\tstd\n'
    
    def getanalyst(start,end):
        result=''
        for j in range(13):
            size=[]
            for i in range(start,end):
                idx=3
                size.append(vector[i,j])
            size=np.array(size)
            result+="%s\t%10.5f\t%10.5f\t%10.5f\t%10.5f"%(name[j],size.min(),size.max(),size.mean(),size.std())
            result+="\n"
        return result


    # allx+=getanalyst(0,2600)
    # allx+='-\t-\t-\t-\t-\n'
    # allx+=getanalyst(2600,3000)
    # with open('./data/xyt/xyt.csv','w') as f:
    #     f.write(allx)
    # print allx
    def draw():
        import matplotlib.pyplot as plt
        for j in range(13):
            size=[]
            for i in range(n):
                idx=3
                size.append(vector[i,j])
            size=np.array(size)
            # result+="%s\t%10.5f\t%10.5f\t%10.5f\t%10.5f"%(name[j],size.min(),size.max(),size.mean(),size.std())
            # result+="\n"        
            plt.plot(range(3000),size)
            plt.title(name[j])
            plt.savefig('./data/xyt/%s.png'%name[j])
            plt.clf()
            # plt.show()
            # exit()
            print j
        # plt.close(0)
        # plt.show()
    draw()

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

    dt=dt if dt>1e-5 else 4200.0
    mt=mt if mt>1e-5 else 700.0
   

    a=dx/dt
    # b=dy/dt
    c=dt/mt
    a= a if abs(a)<1 else 0
    # b= b if abs(b)<1 else 0
    c= c if c<1 else 0
    return [a,c]

def getlast(mouse):
    xn=len(mouse[0])
    mid=xn/2
    idxx=range(xn-6,xn-1)
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
    # b=dy/dt
    c=dt/mt
    a= a if abs(a)<1 else 0
    # b= b if abs(b)<1 else 0
    c= c if c<1 else 0
    return [a,c]


def analyst_xyt2():
    def getxyt(idx,mouse,goal,label):
        tmp=[]
        # a,b,c=getmid(mouse)
        x=getspeed(mouse)
        # print x
        # exit(0)
        for v in x:
            tmp.append(v)
            # print v
        # exit()
            # tmp.append([x)
        # tmp.append(x)
        # tmp.append(t)

        # ex=mouse[0][-1]
        # ey=mouse[1][-1]

        # gx=goal[0]
        # gy=goal[1]
        # distance=(ex-gx)**2+(ey-gy)**2
        # distance=distance**0.5
        # # tmp.append(ey-gy)
        # tmp.append(ey-gy)
        # tmp.append(mouse[0][0])
        # tmp.append(mouse[1][0])
        # tmp.append(mouse[2][-1])
        # xlen=len(mouse[0])
        # tmp.append(xlen) # len 
        # for i in range(3):
        #     tmp.append(mouse[i].min()) # x min
        #     tmp.append(mouse[i].max()) # x max
        #     tmp.append(mouse[i].mean()) # x max
        #     tmp.append(mouse[i].std()) # x max
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

    name=["mouse_size",\
    "x_min","x_max","x_mean","x_std",\
    "y_min","y_max","y_mean","y_std",\
    "t_min","t_max","t_mean","t_std",\
    ]
    allx='name\tmin\tmax\tmean\tstd\n'
    
    def getanalyst(start,end):
        result=''
        for j in range(13):
            size=[]
            for i in range(start,end):
                idx=3
                size.append(vector[i,j])
            size=np.array(size)
            result+="%s\t%10.5f\t%10.5f\t%10.5f\t%10.5f"%(name[j],size.min(),size.max(),size.mean(),size.std())
            result+="\n"
        return result


    # allx+=getanalyst(0,2600)
    # allx+='-\t-\t-\t-\t-\n'
    # allx+=getanalyst(2600,3000)
    # with open('./data/xyt/xyt.csv','w') as f:
    #     f.write(allx)
    # print allx
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
    draw(1)

if __name__=="__main__":
    # analyst_xyt()
    analyst_xyt2()