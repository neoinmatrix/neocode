# coding: utf-8
import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn import  metrics
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import time
import pca

def recognizer(train):
    kf = KFold(n_splits=10, shuffle=True,random_state=np.random.randint(11))
    clf=SVC()
    accuracy=0.0
    confusion=np.zeros([2,2])
    start = time.clock()
    for train_index, test_index in kf.split(train.index):
        X=train.ix[train_index].values
        y=label_train.ix[train_index].values

        clf.fit(X,y)
        # break
        X_test=train.ix[test_index].values
        predicted = clf.predict(X_test)
        expected =label_train.ix[test_index].values
        accy_tmp=metrics.accuracy_score(expected, predicted)
        accuracy+=accy_tmp
        # print "the single accuracy:",accy_tmp

    end = time.clock()
    print "the  final accuracy:%.2f"%(accuracy/10.0)
    print "the time using: %f s" % (end - start)
    
if __name__ == '__main__':
    train=pd.read_csv("mnist.csv")
    # normalize the train data
    label_train=train['label']
    train=train.drop('label', axis=1)
    train = train / 255

    Y,_=pca.pca(train.values.T,0.8,getnew=False)
    train_pca = pd.DataFrame(Y.T.real)
    print '======== data  wiht   pca ================'
    recognizer(train_pca)
    print '======== data without pca ================'
    recognizer(train)
