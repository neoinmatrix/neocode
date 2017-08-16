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
import shutil

def getfeature(idx,mouse,goal,label):
    tmp=[]
    # has changed x towards so must not be machine
    if get_X_PN(mouse)==True:  
        return False
    # the x >437 must be machine 
    if mouse[0][0]>=437:
        return False
    return True
    # return np.array(tmp).reshape([1,len(tmp)])


def main():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
       
    vector_label=[]
    mlist=[]
    for i in range(n):
        tmp=getfeature(0,mouses[i],goals[i],0)
        if tmp==True:
            mlist.append(i+1)
    # print len(mlist)
    path='./data/'
    for v in mlist:
        if v>2600:
            src=path+"pic3d_f/%d.png"%v
            dst=path+"pic3d_f/m/%d.png"%v
        # if v>1000:
        #     print src
        #     print dst
        #     exit()
            shutil.move(src, dst) 
        
       
if __name__=="__main__":
    main()
    pass