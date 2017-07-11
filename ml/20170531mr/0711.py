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

from dealfunc import *


def getfeature(idx,mouse,goal,label):
    tmp=[]
    # has changed x towards so must not be machine
    if get_X_PN(mouse)==True:  
        return False
    # the x >437 must be machine 
    if mouse[0][0]>=437:
        return True

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

def testResultAll(clf,scaler,pca,savepath='./data/0711tmp.txt',stop=1200):
    ds=dataset.DataSet()
    allnum=0
    mclass=[[],[],[]]
    while True:
        idx,mouse,goal,label=ds.readTestFile()
        if idx==False:
            break
        tmp=getfeature(idx,mouse,goal,label)
        if type(tmp) is bool:
            if tmp==False:
                pass
            elif tmp==True:
                mclass[0].append(idx)
                mclass[1].append(idx)
        else:
            tmp=scaler.transform(tmp)
            tmp=pca.transform(tmp)
            r=clf.predict(tmp)
            if r[0]>0:
                pass
            else:
                mclass[0].append(idx)
                mclass[2].append(idx)

        if stop!=-1 and allnum>stop:
            break
        if allnum%1000==0:
            print idx,len(mclass[0]),len(mclass[1]),len(mclass[2])
        allnum+=1
    print "all:",len(mclass[0]),len(mclass[1]),len(mclass[2])
    savestr=createstr(mclass[0])
    with open(savepath,'w') as f:
        f.write(savestr)
    print "ok"


def main():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    vector_label=[]
    for i in range(n):
        tmp=getfeature(0,mouses[i],goals[i],0)
        if type(tmp) is bool:
            continue  
        vector.append(tmp[0])
        vector_label.append(labels[i])

    vector=np.array(vector)
    labels=np.array(vector_label)

    scaler_vector=vector
    vector = preprocessing.scale(vector)
    pca = PCA(n_components=20)
    pca.fit(vector)
    vector=pca.transform(vector)

    dt=datadeal.DataTrain()
    # about 17 w
    clf = MLPClassifier(alpha=0.9,
        activation='logistic', \
        hidden_layer_sizes=(15,19),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )

    # # False
    test=False
    if test==True:
        dt.trainTest(clf,vector,labels,4.0)
    else:
        clf.fit(vector,labels)
        scaler = preprocessing.StandardScaler().fit(scaler_vector)
        testResultAll(clf,scaler,pca,savepath='./data/0711tmp.txt',stop=-1)

if __name__=="__main__":
    main()
    pass