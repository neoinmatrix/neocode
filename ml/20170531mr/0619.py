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

def getvector(idx,mouse,goal,label):
    def getyn(mouse):
        yn=len(mouse[1])
        y=mouse[1]
        if yn==1:
            return [0]
        for i in range(yn)[-1:0:-1]:
            y[i]=y[i]-y[i-1]
        flag=1
        state=y[0]
        ychange=0
        for i in range(1,yn):
            if state*y[i]<0:
                ychange+=1
            state=y[i]
        return [float(ychange)/5.0]
    def getlastx(mouse):
        xn=len(mouse[0])
        x=mouse[0]
        y=mouse[1]
        t=mouse[2]
        sumtall=float(t[xn-1])
        idx=0
        tmp=1.0
        for i in range(1,xn):
            idx=xn-1-i
            tmp=(sumtall-float(t[i]))
            if tmp/sumtall>0.1:
                break
            if tmp==0:
                tmp=1.0
        # print xn-3,xn-2,xn-1,xn,idx
        # print x[xn-3]
        # print x[xn-2]
        # print x[xn-1]
        # print x[idx]
        # print tmp
        xchange=(x[xn-1]-x[idx])
        ychange=(y[xn-1]-y[idx])
        return [abs(xchange)/20,abs(ychange)/20]

    tmp=[]
    tmp.extend([mouse[0][0]/1000,mouse[1][0]/2700])
    # tmp.extend(getendstart(mouse,goal))
    # tmp.extend(startend(mouse))
    tmp.extend(getyn(mouse))
    tmp.extend(getlastx(mouse))

    return np.array(tmp).reshape([1,len(tmp)])
   

def trainTest(clf,X,y):
    kf = KFold(n_splits=10, shuffle=True,random_state=np.random.randint(len(y)))
    accuracy=0.0
    confusion=np.zeros([2,2])
    noget=[]
    nogetb=[]
    for train_index, test_index in kf.split(range(len(y))):
        X_train=X[train_index]
        y_train=y[train_index]
        X_test=X[test_index]
        expected=y[test_index]
        clf.fit(X_train,y_train)
        predicted = clf.predict(X_test)

        for i in range(len(test_index)):
            if expected[i]!=predicted[i] and predicted[i]==1:
                # print test_index[i]
                noget.append(test_index[i])
            if expected[i]!=predicted[i] and predicted[i]==0:
                # print test_index[i]
                nogetb.append(test_index[i])
        print "over"
        # if len(noget)>0:
        #     print test_index
        #     print noget
        #     print nogetb
        #     print len(test_index)
        #     print len(noget)
        #     print len(nogetb)
        #     exit(0)
    return noget
    # print len(noget)
        # print len(predicted)
        # accy_tmp=metrics.accuracy_score(expected, predicted)
        # accuracy+=accy_tmp
    #     conf_tmp=metrics.confusion_matrix(expected, predicted)
    #     confusion+=conf_tmp
    #     print "predited rate:%f"%accy_tmp
    # print confusion
    # print accuracy/10.0

def assemble():
    ds=dataset.DataSet()
    ds.getTrainData()
    # dw=datadraw.DataDraw('2d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    # print getvector(1,mouses[0],goals[0],1)
    # print getvector(1,mouses[1],goals[1],1)
    # print getvector(1,mouses[111],goals[111],1)
    # print getvector(1,mouses[2600],goals[2600],1)
    # print getvector(1,mouses[2700],goals[2700],1)
    # print getvector(1,mouses[2777],goals[2777],1)
    # exit()

    # print mouses[492],goals[492]
    for i in range(n):
        vector.append(getvector(1,mouses[i],goals[i],1)[0])
    vector=np.array(vector)
    # print vector.shape
    # dw =datadraw.DataDraw('2d')
    # dw.drawbatchgoal(vector[0:2600],c='b')
    # dw.drawbatchgoal(vector[2600:],c='r')
    # plt.show()
    dt=datadeal.DataTrain()
    clf = MLPClassifier(alpha=1e-2, hidden_layer_sizes=(20,10), random_state=1)
    # clf = SVC(C=1.65)
    # dt.trainTest(clf,vector,labels)

    dt.train(clf,vector,labels)

    # dt.testResultAll(ds,getvector,savepath='./data/0607tmp.txt')

    dt.testResultAll(ds,getvector,savepath='./data/0619tmp.txt')

    # exit()

    # noget=trainTest(clf,vector,labels)
    # noget=np.array(noget)
    # return noget
    # np.savetxt('./data/tmp0619.txt',noget)
    # numpy.savetxt("filename.txt",a)
    # b =  numpy.loadtxt("filename.txt")

def analystnoget():
    tmp=np.loadtxt('./data/tmp0619.txt')
    tmp=np.array(tmp,dtype='int')
    ds=dataset.DataSet()
    ds.getTrainData()
    dw=datadraw.DataDraw('3d')
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    # vector=[]
    
    colors=dw.getColorsValue()
    count=-1
    dw.draw3dline(mouses[0],c=colors[0])
    dw.draw3dline(mouses[2700],c=colors[15])

    for i in tmp:
        count+=1
        if count==6:
            continue
        dw.draw3dline(mouses[i],c=colors[count%10+1])

    plt.show()



if __name__=="__main__":
    # analystnoget()
    tmp=assemble()
    # tmp=np.loadtxt('./data/tmp0619.txt')
    # tmp=np.array(tmp,dtype='int')
    # print tmp
    # countx=[0]*3000
    # for v in tmp:
    #     countx[v]+=1
    # for i in range(2600,3000):
    #     if countx[i]>0:
    #         print i,countx[i]
    # print "============="
    # for i in range(2600,3000):
    #     if countx[i]==0:
    #         print i,countx[i]
    # print len(tmp)
    pass