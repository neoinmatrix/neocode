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

# def getfeature(idx,mouse,goal,label):
#     tmp=[]
#     f_label=get_spoint_filter(mouse,config)
#     # f_label=get_X_PN(mouse)
#     tmp.append(f_label)
#     # if f_label==0 must be manual tracking 
#     # so set other features with 0 humanity 
#     return np.array(tmp).reshape([1,len(tmp)])
def create(data):
    tmp=''
    for v in data:
        tmp+=str(v)+"\n"
    return tmp

def testResultAll():
    ds=dataset.DataSet()
    allnum=0
    machine=0
    machine_list=[]
    str_record=''
    machine_class=[[],[],[]]
    while True:
        idx,mouse,goal,label=ds.readTestFile()
        if idx==False:
            break
        if get_X_PN(mouse)[0]==0:
            machine_list.append(idx)
            if mouse[0][0]<437:
                machine_class[0].append(idx)
            elif mouse[0][0]==437:
                machine_class[1].append(idx)
            else:
                machine_class[2].append(idx)

        if allnum%1000==0:
            print idx,len(machine_list),\
            len(machine_class[0]),len(machine_class[1]),len(machine_class[2])
        allnum+=1
    print len(machine_list),\
            len(machine_class[0]),len(machine_class[1]),len(machine_class[2])
    str_record=create(machine_class[2])
    with open('./data/0711_th.txt','w') as f:
        f.write(str_record) 

    str_record=create(machine_class[1])
    with open('./data/0711_tm.txt','w') as f:
        f.write(str_record)

    str_record=create(machine_class[0])
    with open('./data/0711_tl.txt','w') as f:
        f.write(str_record)
    # print len(machine_list)
    # print machine_list

def main():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    # vector=[]

    # config["borders"]=get_borders(mouses)
    testResultAll()

if __name__=="__main__":
    main()
    pass