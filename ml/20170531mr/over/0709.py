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


def filter(mouse,f,f_label):
    if f_label==0:
        result=f(mouse)
    return [0]*len(result)

config={'borders':[]}
def getfeature(idx,mouse,goal,label):
    tmp=[]
    f_label_area=get_spoint_filter(mouse,config)
    f_label_normal=get_X_PN(mouse)
    tmp.append(f_label_area+f_label_normal)

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
    vector=[]
    config["borders"]=get_borders(mouses)
    # print config
    # exit()
    for i in range(n):
        vector.append(getfeature(1,mouses[i],goals[i],1)[0])
    vector=np.array(vector)

    scaler_vector=vector
    vector = preprocessing.scale(vector)
    vector = np.c_[scaler_vector[:,0],vector[:,1:]]
    printdemo(vector)
    # print len(vector[0])
    # exit()
    pca = PCA(n_components=1)
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

    print clf
    # clf = MLPClassifier(alpha=1e-4,
    #     activation='logistic', \
    #     hidden_layer_sizes=(16,18),random_state=0,solver='lbfgs',\
    #     max_iter=400)

    # False
    test=True
    if test==True:
        dt.trainTest(clf,vector,labels,4.0)
    else:       
        scaler = preprocessing.StandardScaler().fit(scaler_vector)
        dt.train(clf,vector,labels)
        dt.testResultAll(ds,getfeature,savepath='./data/0706tmp.txt',stop=1200,scal=scaler,pca=pca)
        # dt.testResultAll(ds,getfeature,savepath='./data/0704tmp.txt',stop=1200,scal=scaler)

       
if __name__=="__main__":
    import time
    start =time.clock()
    main()
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))