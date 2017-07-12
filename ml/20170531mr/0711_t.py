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
    # config["borders"]=get_borders(mouses)
    machine_list=[]
    mouse_start=ds.getPosOfMouse(-1)
    mclass=[[],[],[]]
    xyt=0
    for i in range(3000):
        xyt=0
        tmpx=mouse_start[i,xyt]-goals[i][xyt]
        xyt=1
        tmpy=mouse_start[i,xyt]-goals[i][xyt]
        tmp=(tmpx**2+tmpy**2)**0.5
        if tmp>3000:
            tmp=3000
        # if tmp<-60:
        #     tmp=-60
        machine_list.append(tmp)

    value=np.array(machine_list)
    dw=datadraw.DataDraw(typex='2d')
    # dw.draw3dline(mouses[1802])
    # print goals.shape
    dw.drawgoal([range(2600),value[:2600]],'b')
    dw.drawgoal([range(2600,3000),value[2600:]],'r')

    plt.show()
    print value.min()
    print value.max()
    print value.mean()

    # mouse_start=ds.getPosOfMouse(0)
    #     if abs(mouse_start[i,xyt]<goals[i][xyt]):
    #         mclass[0].append(i)
    #     elif mouse_start[i,xyt]==goals[i][xyt]:
    #         mclass[1].append(i)
    #     else:
    #         mclass[2].append(i)
    # for i in range(3):
    #     print len(mclass[i])
    # print mclass[0]
    # print mclass[1]
    # print mouse_start.shape
    # print mouse_start.shape
    # ms=mouse_start[:,0]
    # for i in range(2600):
    #     if ms[i][0]==437:
    #         print i,ms[i][0]
       
if __name__=="__main__":
    main()
    pass