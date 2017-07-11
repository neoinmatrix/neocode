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


config={'borders':[]}

def dealmouse(mouse):
    t=mouse[2]
    n=len(t)
    item=0
    for i in range(1,n):
        if t[i-1]>t[i]:
            # print i 
            item+=1
            break
    return item 

def getfeature(idx,mouse,goal,label):
    tmp=[]
    f_label=get_spoint_filter(mouse,config)
    f_label=get_X_PN(mouse)

    tmp.append(f_label)

    # if f_label==0 must be manual tracking 
    # so set other features with 0 humanity 
    return np.array(tmp).reshape([1,len(tmp)])

def main():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    config["borders"]=get_borders(mouses)
    machine_list=[]
    for i in range(n):
        mouse=mouses[i]
        f_label_a=get_spoint_filter(mouse,config)[0]
        f_label_b=get_X_PN(mouse)[0]
        print f_label_a,f_label_b,i
        if f_label_a==0 and f_label_b==0:
            machine_list.append(i)
    print len(machine_list)


    # def getfeature(idx,mouse,goal,label):
    #     tmp=[]
    #     f_label=get_spoint_filter(mouse,config)
    #     # f_label=get_X_PN(mouse)
    #     tmp.append(f_label)

    # idxs=[]
    # for i in range(n):
    #     cnum=dealmouse(mouses[i])
    #     if cnum>0:
    #         print i,cnum
    #         idxs.append(i) 


    # dw=datadraw.DataDraw(typex='3d')
    # dw.draw3dline(mouses[1802])
    # # print goals.shape
    # # dw.drawgoal([mse[:2600,0],mse[:2600,1]],'b')
    # # dw.drawgoal([mse[2600:,0],mse[2600:,1]],'r')
    # # for i in range(2600):
    # #   dw.draw3dgoal(mse[i],c='b')
    # # for i in range(2600,3000):
    # #   dw.draw3dgoal(mse[i],c='r')
    # plt.show()

    #     vector.append(getfeature(1,mouses[i],goals[i],1)[0])
    # vector=np.array(vector)

    
       
if __name__=="__main__":
    main()
    pass