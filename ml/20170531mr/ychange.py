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

def dealmouse(mouse):
    t=mouse[0]
    n=len(t)
    item=0
    for i in range(1,n):
        if t[i-1]>t[i]:
            # print i 
            item+=1
    return item 

def testResultAll():
    ds=dataset.DataSet()
    allnum=0
    machine=0
    machine_list=[]
    while True:
        idx,mouse,goal,label=ds.readTestFile()
        if idx==False:
            break
        if dealmouse(mouse)>0:
            machine_list.append(idx)

        if allnum%1000==0:
            print idx,machine
        allnum+=1
    print len(machine_list)
    # print machine_list

def main():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    idxs=[]
    # print n
    for i in range(n):
        cnum=dealmouse(mouses[i])
        if cnum>0:
            print i,cnum
            idxs.append(i) 
    print len(idxs)

    dw=datadraw.DataDraw(typex='3d')
    # dw.draw3dline(mouses[2600],c='r')
    # dw.draw3dline(mouses[2700],c='g')
    dw.draw3dline(mouses[2800],c='b')
    dw.draw3dline(mouses[2870],c='g')
    dw.draw3dline(mouses[2900],c='y')
    # dw.draw3dline(mouses[2999],c='r')
    # print goals.shape
    # dw.drawgoal([mse[:2600,0],mse[:2600,1]],'b')
    # dw.drawgoal([mse[2600:,0],mse[2600:,1]],'r')
    # for i in range(2600):
    #   dw.draw3dgoal(mse[i],c='b')
    # for i in range(2600,3000):
    #   dw.draw3dgoal(mse[i],c='r')
    plt.show()

    #     vector.append(getfeature(1,mouses[i],goals[i],1)[0])
    # vector=np.array(vector)

    
       
if __name__=="__main__":
    # testResultAll()
    main()
    # end = time.clock()
    # print('Running time: %s Seconds'%(end-start))

    # print datadeal.calcScoreRerve(0.9480,20045)
    pass