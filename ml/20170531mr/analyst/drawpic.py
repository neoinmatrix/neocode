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
    dw=datadraw.DataDraw(typex='3d')
    path='./data/pic/'
    for i in range(n):
        if i<2000:
            continue
        # if i>3000:
        #     break
        fig = plt.figure()  
        ax = fig.add_subplot(111, projection='3d') 
        dw.draw3dline(mouses[i],ax,c='g')
        plt.savefig(path+"%d.png"%i)
        plt.clf()
        plt.close()
        # if i >3:
        #     break
        if i%20==0:
            print i
        # break
    print "over"
    # plt.show()

       
if __name__=="__main__":
    main()
    pass