# coding=utf-8
import dataset
import datadeal
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import  metrics

class DataTrain:
    clf=''

    def train(self,clf,X,y):
        clf.fit(X,y)
        self.clf=clf

    def trainTest(self,clf,X,y):
        kf = KFold(n_splits=10, shuffle=True,random_state=np.random.randint(len(y)))
        accuracy=0.0
        for train_index, test_index in kf.split(range(len(y))):
            X_train=X[train_index]
            y_train=y[train_index]
            X_test=X[test_index]
            expected=y[test_index]
            clf.fit(X_train,y_train)
            predicted = clf.predict(X_test)
            accy_tmp=metrics.accuracy_score(expected, predicted)
            accuracy+=accy_tmp
            print "predited rate:%f"%accy_tmp
        print accuracy/10.0

    def getResult(self,X):
        return self.clf.predict(X)

    def testResultAll(self,ds,f,savepath='./data/result.txt'):
        allnum=0
        machine=0
        machine_list=[]
        while True:
            idx,mouse,goal,label=ds.readTestFile()
            if idx==False:
                break
            item=f(idx,mouse,goal,label)
            r=self.getResult(item)
            allnum+=1
            if allnum%1000==0:
                print idx,machine
            if r>0:
                continue
            else:
                machine+=1
                machine_list.append(idx)

        print "all machine data:%d"%machine
        result=''
        for v in machine_list:
            result+='%s\n'%v
        with open(savepath,'w') as f:
            f.write(result)
        print "ok"

def calcScore(rm,jm,rms=20000.0):
    P=rm/jm
    R=rm/rms
    s=(5*P*R)/(2*P+3*R)
    return s
    
def calcScoreRerve(s,jm,rms=20000.0):
    x=s*(2*rms+3*jm)/5.0
    return x

if __name__=="__main__":
    pass
    


