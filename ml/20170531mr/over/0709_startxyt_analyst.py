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
    # f_label=get_X_PN(mouse)

    tmp.append(f_label)

    # if f_label==0 must be manual tracking 
    # so set other features with 0 humanity 


    return np.array(tmp).reshape([1,len(tmp)])

def testResultAll():
    ds=dataset.DataSet()
    allnum=0
    machine=0
    machine_list=[]
    str_record=''
    while True:
        idx,mouse,goal,label=ds.readTestFile()
        if idx==False:
            break
        # if idx==2:
        #     print config
        #     print get_spoint_filter(mouse,config)
        #     print mouse[0][0],mouse[1][0],mouse[2][0]
        #     exit()
        if get_X_PN(mouse)[0]==0:
            # if get_spoint_filter(mouse,config)[0]==0:
                machine_list.append(idx)
                str_record+=str(idx)+"\n"
            # if dealmouse(mouse)==0:
        # if idx==5:
        #     print machine_list
        #     exit(0) 
        if allnum%1000==0:
            print idx,len(machine_list)
        allnum+=1
    with open('./data/0709_2f.txt','w') as f:
        f.write(str_record)
    print len(machine_list)
    # print machine_list

def main():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    # vector=[]

    config["borders"]=get_borders(mouses)
    testResultAll()
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
    # testResultAll()
    # def cal(a,b):
    #     return float(a)/float(a+b)

    # # print(cal(386,24))
    # # print(cal(386,15))
    # # print(cal(382,29))
    # import time
    # start =time.clock()
    main()
    # print datadeal.calcScoreRerve(0.9973,20000)
    # end = time.clock()
    # print('Running time: %s Seconds'%(end-start))

    # print datadeal.calcScoreRerve(0.9480,20045)
    pass