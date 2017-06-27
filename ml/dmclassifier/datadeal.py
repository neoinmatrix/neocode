# coding=utf-8
import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import  metrics
class DataDeal:
    def getPima(self):
        train=pd.read_csv("./data/pima.csv",header=None)
        train_normal=(train-train.min())/(train.max()-train.min())
        target=np.array(train[8].values,dtype=np.int)
        train_normal=train_normal.drop(8,axis=1)
        data=np.array(train_normal.values,dtype=np.float)
        nums=len(data[:,0]) # pima has x numbers items
        types=2 # pima has two types 1 2
        return  data,target,nums,types
    def getIris(self):
        train=pd.read_csv("./data/iris.csv",header=None)
        train_normal=(train-train.min())/(train.max()-train.min())
        target=np.array(train[4].values,dtype=np.int)
        train_normal=train_normal.drop(4,axis=1)
        data=np.array(train_normal.values,dtype=np.float)
        nums=len(data[:,0]) # iris has x numbers items
        types=3 # iris has two types 1 2 3
        return data,target,nums,types
    def test(self,clf,data,target,nums,types,isprint=False):
        kf = KFold(n_splits=10, shuffle=True,random_state=np.random.randint(11))
        aprfc={'acc':0.0,'p':0.0,'r':0.0,'f1':0.0,'confusion':np.zeros([types,types])}
        aprfc_f={'acc':metrics.accuracy_score,\
        'p':metrics.precision_score,'r':metrics.recall_score,\
        'f1':metrics.f1_score,'confusion':metrics.confusion_matrix}
        for train_index, test_index in kf.split(range(nums)):
            X=data[train_index]
            y=target[train_index]
            clf.fit(X,y)
            X_test=data[test_index]
            y_expected = target[test_index]
            predicted = clf.predict(X_test)
            for p in aprfc:
                if p in ['f1','p','r']:
                    aprfc[p]+=aprfc_f[p](y_expected, predicted, average='macro')
                elif p=='confusion':
                    aprfc[p]+=aprfc_f[p](y_expected, predicted, labels=range(types))
                else:
                    aprfc[p]+=aprfc_f[p](y_expected, predicted)
            if isprint:
                print aprfc
        return aprfc
    def plot_correlation(self,cm, genre_list, name, title):
        from matplotlib import pylab 
        pylab.clf()
        pylab.matshow(cm, fignum=False, cmap='Greens', vmin=0, vmax=1.0)
        ax = pylab.axes()
        ax.set_xticks(np.arange(0,len(genre_list),1))
        ax.set_xticklabels(genre_list)
        ax.xaxis.set_ticks_position("bottom")
        ax.set_yticks(np.arange(0,len(genre_list),1))
        ax.set_yticklabels(genre_list)
        pylab.title(title)
        pylab.colorbar()
        # pylab.grid(True)
        pylab.show()
        pylab.xlabel('x class')
        pylab.ylabel('y class')
        pylab.grid(True)
if __name__=="__main__": 
    dd=DataDeal()
    pima=dd.getPima()
    data,target,nums,types=pima
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    result=dd.test(clf,data,target,nums,types)
    print "=========pima\n", result
    # iris=dd.getIris()
    # data,target,nums,types=iris
    # p,r,t = metrics.precision_recall_curve(y_expected, predicted)
