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

def main():
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

    scaler_vector=vector
    vector = preprocessing.scale(vector)
    vector = np.c_[scaler_vector[:,0],vector[:,1:]]
    # printsomething(vector)

    pca = PCA(n_components=13)
    pca.fit(vector)
    vector=pca.transform(vector)

    dt=datadeal.DataTrain()
    # about 17 w
    clf = MLPClassifier(alpha=0.9,
        activation='logistic', \
        hidden_layer_sizes=(39),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )

    print clf
    # clf = MLPClassifier(alpha=1e-4,
    #     activation='logistic', \
    #     hidden_layer_sizes=(16,18),random_state=0,solver='lbfgs',\
    #     max_iter=400)

    # False
    test=False
    if test==True:
        dt.trainTest(clf,vector,labels,4.0)
    else:       
        scaler = preprocessing.StandardScaler().fit(scaler_vector)
        dt.train(clf,vector,labels)
        dt.testResultAll(ds,getfeature,savepath='./data/0704tmp.txt',stop=-1,scal=scaler,pca=pca)
        # dt.testResultAll(ds,getfeature,savepath='./data/0704tmp.txt',stop=1200,scal=scaler)

       
if __name__=="__main__":
    def cal(a,b):
        return float(a)/float(a+b)

    # print(cal(386,24))
    # print(cal(386,15))
    # print(cal(382,29))
    import time
    start =time.clock()
    main()
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))

    # print datadeal.calcScoreRerve(0.9480,20045)
    pass