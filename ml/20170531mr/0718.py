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

from dealfunc18 import *

def get_t_T(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    x=x/x.max()
    y=y/y.max()
    t=t/t.max()
    xn=len(mouse[0])

    arr3d=[0.0]
    for i in range(0,xn):
        if i+1>=xn:
            break
        else:
            vx1=x[i+1]-x[i]
            dt=t[i+1]-t[i]
            if dt==0:
                continue
            angle=vx1/dt
            arr3d.append(angle)
    tmp=[]
    # 10 point numbers
    beginfive=getfivex(arr3d,range(0,5))
    tmp.extend(beginfive)
    lastfive=getfivex(arr3d,range(len(arr3d)-5,len(arr3d)))
    tmp.extend(lastfive)
    # the first five point  in all percent
    trate=sum(t[:5])/abs(t[-1])
    tmp.append(trate)
    # get 3 of first time 
    tget=getfivex(t,range(0,3),3)
    tmp.extend(tget)
    # judge the second point if large than 500ms
    if  len(mouse[2])>1:
        tmp.append(1.0 if mouse[2][1]>500 else 0.0)
    else:
        tmp.append(0.0)
    # first five angle is concave convex
    xfive=getfivex(x,range(0,5))
    tfive=getfivex(t,range(0,5))
    ax=xfive[4]-xfive[0]
    at=xfive[4]-tfive[0]
    mx=(xfive[4]+xfive[0])/2
    mt=(tfive[4]+tfive[0])/2
    bx=mx-xfive[2]
    bt=mt-tfive[2]
    angle=ax*bx+at*bt
    tmp.append(angle)

    return tmp

def getfeature(idx,mouse,goal,label,use_all=False):
    tmp=[]
    if use_all==True:
        pass
    else:
        # has changed x towards so must not be machine
        if get_X_PN(mouse)==True:  
            return False
        
    anglenum=get_distribution_3dangle(mouse)
    tmp.extend(anglenum)
    return np.array(tmp)

def getfeature2(idx,mouse,goal,label,use_all=False):
    tmp=[]
    if use_all==True:
        pass
    else:
        # has changed x towards so must not be machine
        if get_X_PN(mouse)==True:  
            return False
        # the x >437 must be machine 
        # if mouse[0][0]>=437:
        #     return True
        # sangle=get_sharp_angle(mouse)
        # if sangle==True:
        #     return True
    lastfive=get_t_T(mouse)
    tmp.extend(lastfive)
    return np.array(tmp)

def draw_analyst(idx,mouse,goal,label):
    pass
    notfind=[120,122,129,146,268,276]
    nmachine=[146,268,276]
    path='./data/18/notfind/'
    if idx in notfind:
        print idx
        if idx in nmachine:
            c='b'
        else:
            c='g' 
        fig = plt.figure()  
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(mouse[0],mouse[1],mouse[2],c=c)
        ax.scatter(mouse[0],mouse[1],mouse[2],c=c)
        plt.title(str(idx))
        plt.savefig(path+"3d_%3d.png"%idx)
        plt.clf()
        plt.close()

    if idx in notfind:
        if idx in nmachine:
            c='b'
        else:
            c='g' 
        plt.plot(mouse[0],mouse[1],c=c)
        plt.scatter(mouse[0],mouse[1],c=c)
        plt.title(str(idx))
        plt.savefig(path+"2d/3d_%3d.png"%idx)
        plt.clf()
        plt.close()

    if idx>300:
        exit()

def draw_analyst_single(idx,mouse,goal,label):
    pass
    # notfind=[1,5,45,50,55,57,58,62,63,65,98,99,100,111,132,145,170,182,183,202,226,232,260,266,304,353,356,362,365,371,373,380,390,394,396,423,457,468,477,484,490,504,506,509,534,548,606,630,654,659,671,680,691,693,695,701,705,717,738,743,748,751,753,757,760,772,781,790,801,816,832,836,842,870,879,886,888,907,920,921,941,968,974,989,990,993,997]
    notfind=[120,122,129,46,268,276]
    nmachine=[46,268,276]
    path='./data/17/notfind/'
    if idx in notfind:
        print idx
        if idx in nmachine:
            c='b'
        else:
            c='g' 
        fig = plt.figure()  
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(mouse[0],mouse[1],mouse[2],c=c)
        ax.scatter(mouse[0],mouse[1],mouse[2],c=c)
        plt.title(str(idx))
        plt.show()
        plt.clf()
        plt.close()

    if idx>300:
        exit()

def testResult(config={}):
    scaler=config["scaler"]
    clf=config["clf"]
    pca=config["pca"]
    savepath=config["savepath"]
    stop=config["stop"]
    scaler_extra=config["scaler_line"]
    clf_extra=config["clf_line"]

    ds=dataset.DataSet(testfp='/data/dsjtzs_txfz_testB.txt')
    allnum=0
    mclass=[[],[],[],[],[],[],[],[]]
    rclass=[[],[],[],[],[],[],[],[]]
    np.set_printoptions(formatter={'float':lambda x: "%5.5f"%float(x)})
    while True:
        allnum+=1
        if stop!=-1 and allnum>stop:
            break
        if allnum%1000==0:
            tmp1=''
            for i in range(len(mclass)):
                if len(mclass[i])>0:
                    tmp1+="%d "%len(mclass[i])
            print allnum
            print tmp1
            tmp2=''
            for i in range(len(rclass)):
                if len(rclass[i])>0:
                    tmp2+="%d "%len(rclass[i])
            print tmp2
            print "======"

        idx,mouse,goal,label=ds.readTestFile()
        if idx==False:
            break
        # draw_analyst_single(idx,mouse,goal,label)
        # draw_analyst(idx,mouse,goal,label)

        # the x >=437 must be machine 
        if mouse[0][0]>=437:
            mclass[0].append(idx)
            mclass[1].append(idx)
            continue

        # exists acute angle must be machine 
        sangle=get_sharp_angle(mouse)
        if sangle==True:
            mclass[0].append(idx)
            mclass[2].append(idx)
            continue

        tmp=getfeature(idx,mouse,goal,label)
        if type(tmp) is bool and tmp==False:
            mclass[3].append(idx)
            continue

        # the special tracking
        rclass[0].append(idx)
        tmp=scaler.transform([tmp])
        r=clf.predict(tmp)
        
        if r[0]==1:
            rclass[1].append(idx)
            # this class is other line type 
            tmp=getfeature2(idx,mouse,goal,label)
            tmp=scaler_extra.transform([tmp])
            rr=clf_extra.predict(tmp)
            if rr[0]==0:
                mclass[0].append(idx)
                rclass[2].append(idx)
            pass
        else:
            # this class is L or \ line type
            rclass[3].append(idx)
            tmp=getfeature2(idx,mouse,goal,label)
            tmp=scaler_extra.transform([tmp])
            rr=clf_extra.predict(tmp)
            if rr[0]==0:
                # this is according dx/dt find machine
                mclass[0].append(idx)
                rclass[4].append(idx)
            else:
                # this is not machine
                rclass[5].append(idx)   

    getSummary(mclass,rclass,savepath)
    print "ok"

def maintest():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    mouses_tmp=[]
    labels_tmp=[]    
    mouses_tmp2=[]
    labels_tmp2=[]
 
    for i in range(n):
        tmp=getfeature(i,mouses[i],goals[i],0,use_all=True)
        if type(tmp) is bool:
            # print i,tmp
            continue
        if i in range(0,2600):
            continue
        if i in range(2600,2650):
            labels_tmp.append(1)
        elif i in range(2650,2700):
            labels_tmp.append(2)
        elif i in range(2700,2800):
            labels_tmp.append(1)
        elif i in range(2800,2900):
            labels_tmp.append(2)
        else: 
            labels_tmp.append(1)
        mouses_tmp.append(tmp)

    for i in range(n):
        tmp=getfeature2(i,mouses[i],goals[i],0,use_all=False)
        if type(tmp) is bool:
            # print i,tmp
            continue
        labels_tmp2.append(labels[i])
        mouses_tmp2.append(tmp)

    labels=np.array(labels_tmp)
    vector=np.array(mouses_tmp)
    labels2=np.array(labels_tmp2)
    vector2=np.array(mouses_tmp2)
    scaler_vector=vector
    scaler_vector2=vector2
    # for i in range(20):
    #     plt.plot(range(7),vector2[i],c='g')
    # for i in range(1300,1320):
    #     plt.plot(range(7),vector2[i],c='b')
    # plt.show()
    # print vector2.shape
    # exit()
    np.set_printoptions(formatter={'float':lambda x: "%5.3f"%float(x)})
    vector = preprocessing.scale(vector)
    vector2 = preprocessing.scale(vector2)
    # pca = PCA(n_components=9)
    # pca.fit(vector)
    # vector=pca.transform(vector)

    dt=datadeal.DataTrain()
    clf = MLPClassifier(alpha=0.5,
        activation='logistic', \
        hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    ) 
    clf2 = MLPClassifier(alpha=0.2,
        activation='logistic', \
        hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )
    clf.fit(vector,labels)
    clf2.fit(vector2,labels2)
    scaler = preprocessing.StandardScaler().fit(scaler_vector)
    scaler2 = preprocessing.StandardScaler().fit(scaler_vector2)
    config={
        "scaler":scaler,
        "clf":clf,
        "pca":'',
        "savepath":'./data/18b/',
        "stop":-1,
        "scaler_line":scaler2,
        "clf_line":clf2,
    }
    testResult(config)

def maintrain():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    labels_tmp=[]
    mouses_tmp=[]
    mouses_tmp2=[]
    labels_tmp2=[]
    for i in range(n):
        tmp=getfeature(i,mouses[i],goals[i],0,use_all=True)
        # tmp=getfeaturetest(i,mouses[i],goals[i],0,use_all=False)
        if type(tmp) is bool:
            # print i,tmp
            continue
        if i in range(0,2600):
            continue
        if i in range(2600,2650):
            # labels_tmp.append(1)
            labels_tmp.append(1)
        elif i in range(2650,2700):
            # labels_tmp.append(2)
            labels_tmp.append(2)
            # labels_tmp.append(2)
        elif i in range(2700,2800):
            labels_tmp.append(1)
        elif i in range(2800,2900):
            labels_tmp.append(2)
        else: 
            labels_tmp.append(1)
        mouses_tmp.append(tmp)

    for i in range(n):
        tmp=getfeature2(i,mouses[i],goals[i],0,use_all=False)
        if type(tmp) is bool:
            continue
        labels_tmp2.append(labels[i])
        mouses_tmp2.append(tmp)

    labels2=np.array(labels_tmp2)
    vector2=np.array(mouses_tmp2)
    labels=np.array(labels_tmp)
    vector=np.array(mouses_tmp)
    scaler_vector=vector
    scaler_vector2=vector2
    # for i in range(2000,2050):
    #     print vector[i]
    #     plt.plot(range(5),vector[i],c='b')
    # for i in range(2600,2650):
    #     print vector[i]
    #     plt.plot(range(5),vector[i],c='g')
    # plt.show()
    # exit(0)
    # print vector.shape
    # exit()
    # plt.scatter(range(len(vector)),vector[:,1])
    # plt.show()
    # vector = preprocessing.scale(vector)
    vector2 = preprocessing.scale(vector2)
    # pca = PCA(n_components=9)
    # pca.fit(vector)
    # vector=pca.transform(vector)

    dt=datadeal.DataTrain()
    clf2 = MLPClassifier(alpha=0.2,
        activation='logistic', \
        hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )
    np.set_printoptions(formatter={'float':lambda x: "%d"%float(x)})
    # confusion=dt.trainTest(clf,vector,labels,4.0,classn=6,returnconfusion=True)
    confusion=dt.trainTest(clf2,vector2,labels2,4.0,classn=2,returnconfusion=True)
    # dw=datadraw.DataDraw('2d')
    # confusion=confusion**0.1
    # datadraw.plot_confusion_matrix(confusion,range(6),'a','a',max=confusion.max())

def main():
    # maintrain()   
    maintest()

if __name__=="__main__":
    main()
    pass
