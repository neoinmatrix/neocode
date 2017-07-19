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

def getf(mouse):
    tmp=[]
    t=mouse[2]
    tmp.append(t[0])
    angles=getfivex(t,range(5),5)
    angles=np.array(angles)
    tmp.append(angles.std())
    tmp.append(angles.mean())
    return np.array(tmp)

    # for i in range(1,5):
    #     dt=angles[i]-angles[i-1]
    #     tmp.append(dt*10)
    # return np.array(tmp)

def getft(mouse):
    tmp=[]
    t=mouse[2]
    dt=[0.0]
    for i in range(len(t)-1):
        dt.append(t[i+1]-t[i]) 
    dt=np.array(dt)
    tmp.append(dt.max())
    tmp.append(dt.std())
    tmp.append(dt.mean())
    return np.array(tmp)

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
    lastfive=getfivex(arr3d,range(len(arr3d)-5,len(arr3d)))
    tmp.extend(lastfive)
    beginfive=getfivex(arr3d,range(0,5))
    tmp.extend(beginfive)
    # the first five point  in all percent
    trate=sum(t[:5])/abs(t[-1])
    tmp.append(trate)
    # get 3 of first time 
    t3=getfivex(t,range(0,3),3)
    tmp.extend(t3)
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

def getfeature(idx,mouse,goal,label):
    tmp=[]
    anglenum=get_distribution_3dangle(mouse)
    tmp.extend(anglenum)
    return np.array(tmp)

def getfeature2(idx,mouse,goal,label):
    tmp=[]
    lastfive=get_t_T(mouse)
    tmp.extend(lastfive)
    return np.array(tmp)

def draw_analyst(idx,mouse,goal,label,notfind='',nmachine='',path='',stop=1000):
    if notfind=='':
        notfind=[17,19,32,47,53,59,64,67,78,84,86,87,89,92,101,113,118,129,130,137,138,150,154,169,172,176,184,210,214,223,236,237,240,252,259,265,268,282,287,288,301,330,348,358,363,372,381,400,403,427,429,439,440,450,452,466,470,473,488,501,513,528,533,550,553,557,561,564,567,574,589,590,594,605,610,611,613,617,626,637,656,660,661,665,681,686,688,700,709,716,726,737,750,766,768,770,779,780,791,793,794,806,807,818,821,834,844,846,865,871,877,880,884,899,913,916,918,931,934,940,942,944,945,949,955,956,958,965,967,976,985,992]
    if nmachine=='': 
        nmachine=[17,19,32,47,53,59,64,67,78,84,86,87,89,92,101,113,118,129,130,137,138,150,154,169,172,176,184,210,214,223,236,237,240,252,259,265,282,287,288,301,330,348,358,363,372,381,400,403,427,429,439,440,452,470,473,488,513,528,533,550,553,557,561,564,567,574,589,590,594,605,610,611,613,617,626,637,656,660,661,665,681,686,688,700,709,716,726,737,750,766,770,779,780,791,793,794,806,807,818,821,834,844,846,865,871,877,880,884,899,913,916,918,931,934,940,942,944,945,949,955,956,958,965,967,976,985,992]
    if path=='':
        path='./data/19/notfind/'
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

    if idx>stop:
        exit()

def draw_analyst_single(idx,mouse,goal,label,notfind='',nmachine='',path='',stop=1000,save=False):
    if notfind=='':
        notfind=[17,19,32,47,53,59,64,67,78,84,86,87,89,92,101,113,118,129,130,137,138,150,154,169,172,176,184,210,214,223,236,237,240,252,259,265,268,282,287,288,301,330,348,358,363,372,381,400,403,427,429,439,440,450,452,466,470,473,488,501,513,528,533,550,553,557,561,564,567,574,589,590,594,605,610,611,613,617,626,637,656,660,661,665,681,686,688,700,709,716,726,737,750,766,768,770,779,780,791,793,794,806,807,818,821,834,844,846,865,871,877,880,884,899,913,916,918,931,934,940,942,944,945,949,955,956,958,965,967,976,985,992]
    if nmachine=='': 
        nmachine=[17,19,32,47,53,59,64,67,78,84,86,87,89,92,101,113,118,129,130,137,138,150,154,169,172,176,184,210,214,223,236,237,240,252,259,265,282,287,288,301,330,348,358,363,372,381,400,403,427,429,439,440,452,470,473,488,513,528,533,550,553,557,561,564,567,574,589,590,594,605,610,611,613,617,626,637,656,660,661,665,681,686,688,700,709,716,726,737,750,766,770,779,780,791,793,794,806,807,818,821,834,844,846,865,871,877,880,884,899,913,916,918,931,934,940,942,944,945,949,955,956,958,965,967,976,985,992]
    if path=='':
        path='./data/19/notfind/'
    if idx in notfind:
        print idx
        if idx in nmachine:
            c='b'
        else:
            c='g' 
        fig = plt.figure()  
        ax = fig.add_subplot(221, projection='3d')
        ax.plot(mouse[0],mouse[1],mouse[2])
        ax.scatter(mouse[0],mouse[1],mouse[2])
        ax = fig.add_subplot(222)
        ax.plot(mouse[0],mouse[1],c='r')
        ax.scatter(mouse[0],mouse[1],c='r')
        ax = fig.add_subplot(223)
        ax.plot(mouse[0],mouse[2],c='g')
        ax.scatter(mouse[0],mouse[2],c='g')
        ax = fig.add_subplot(224)
        ax.plot(mouse[1],mouse[2],c='b')
        ax.scatter(mouse[1],mouse[2],c='b')
        plt.title(str(idx)) 
        if save==True:
            plt.title(str(idx))
            plt.savefig(path+"3f_%4d.png"%idx)
            plt.clf()
            plt.close()
        else:
            plt.show()
    if idx>1000:
        exit()

def testResult(config={}):
    scaler=config["scaler"]
    clf=config["clf"]
    pca=config["pca"]
    savepath=config["savepath"]
    stop=config["stop"]
    scaler_extra=config["scaler_line"]
    clf_extra=config["clf_line"]
    scaler_x437=config["scaler_x437"]
    clf_x437=config["clf_x437"]    

    scaler_sharp=config["scaler_sharp"]
    clf_sharp=config["clf_sharp"]

    # ds=dataset.DataSet(testfp='./data/dsjtzs_txfz_testB.txt')
    ds=dataset.DataSet()
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

        draw_analyst_single(idx,mouse,goal,label,\
            notfind=[190,370,419,777],nmachine=[83,156])
        # draw_analyst(idx,mouse,goal,label)

        # has changed x towards so must not be machine
        if get_X_PN(mouse)==True:
            continue

        # the x >=437 must be machine 
        if mouse[0][0]>=437:
            tmp=getf(mouse)
            tmp=scaler_x437.transform([tmp])
            rr=clf_x437.predict(tmp)
            if rr[0]==1:
                mclass[1].append(idx)
            else:
                mclass[2].append(idx)
                mclass[0].append(idx)
            continue

        # exists acute angle must be machine 
        sangle=get_sharp_angle(mouse)
        if sangle==True:
            mclass[4].append(idx)
            mclass[0].append(idx)
            # tmp=getft(mouse)
            # tmp=scaler_sharp.transform([tmp])
            # rr=clf_sharp.predict(tmp)
            # if rr[0]==1:
            #     mclass[3].append(idx)
            # else:
            #     mclass[4].append(idx)
            #     mclass[0].append(idx)
            # continue

        tmp=getfeature(idx,mouse,goal,label)

        # the special tracking
        rclass[0].append(idx)
        tmp=scaler.transform([tmp])
        r=clf.predict(tmp)
        
        if r[0]==1: # this class is other line type 
            rclass[1].append(idx)
            tmp=getfeature2(idx,mouse,goal,label)
            tmp=scaler_extra.transform([tmp])
            rr=clf_extra.predict(tmp)
            if rr[0]==0:
                mclass[0].append(idx)
                rclass[2].append(idx)
            continue
        else:   # this class is L or \ line type
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
    mouses_x437=[]
    labels_x437=[]   
    mouses_sharp=[]
    labels_sharp=[]
 
    for i in range(n):
        tmp=getfeature(i,mouses[i],goals[i],0)
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
        if get_X_PN(mouses[i])==True:
            continue
        tmp=getfeature2(i,mouses[i],goals[i],0)
        mouses_tmp2.append(tmp)
        labels_tmp2.append(labels[i])
    
    for i in range(n):
        tmp=get_X_PN(mouses[i])
        if mouses[i][0][0]>=437 and tmp==False:
            mouses_x437.append(getf(mouses[i]))
            labels_x437.append(labels[i])
        if get_sharp_angle(mouses[i])==True:
            # print i
            mouses_sharp.append(getft(mouses[i]))
            labels_sharp.append(labels[i])

    labels=np.array(labels_tmp)
    vector=np.array(mouses_tmp)
    labels2=np.array(labels_tmp2)
    vector2=np.array(mouses_tmp2)
    labels_x437=np.array(labels_x437)
    vector_x437=np.array(mouses_x437)    
    labels_sharp=np.array(labels_sharp)
    vector_sharp=np.array(mouses_sharp)

    scaler_vector=vector
    scaler_vector2=vector2
    scaler_vector_x437=vector_x437
    scaler_vector_sharp=vector_sharp
    # print vector_sharp.shape
    # exit()

    np.set_printoptions(formatter={'float':lambda x: "%5.3f"%float(x)})
    vector = preprocessing.scale(vector)
    vector2 = preprocessing.scale(vector2)
    vector_x437 = preprocessing.scale(vector_x437)
    vector_sharp = preprocessing.scale(vector_sharp)
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
    # clf_x437 = MLPClassifier(alpha=1.2,
    #     activation='logistic', \
    #     hidden_layer_sizes=(3,5),random_state=0,solver='lbfgs',\
    #     max_iter=250,early_stopping=True, epsilon=1e-04,\
    #     # learning_rate_init=0.1,learning_rate='invscaling',
    # )

    #best 275 35  a=1.0 size 11,11 
    #best 339 68  a=0.8 size 11,11 
    #best 366 74  a=0.6 size 11,11 
    #best 421 116  a=0.6 size 11,11 
    #best 421 116  a=0.6 size 13,13 
    #best 433 133  a=0.6 size 9,13 
    #best 433 133  a=0.6 size 13,9 
    #best 413 110  a=0.6 size 13,7
    clf_x437 = MLPClassifier(alpha=0.6,
        activation='logistic', \
        hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )

    clf_sharp = MLPClassifier(alpha=0.2,
        activation='logistic', \
        hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )

    # clf_x437=SVC(C=2.2)
    # clf2=SVC(C=1.2)
    clf.fit(vector,labels)
    clf2.fit(vector2,labels2)
    clf_x437.fit(vector_x437,labels_x437)
    # clf_sharp.fit(vector_sharp,labels_sharp)

    scaler = preprocessing.StandardScaler().fit(scaler_vector)
    scaler2 = preprocessing.StandardScaler().fit(scaler_vector2)
    scaler_x437 = preprocessing.StandardScaler().fit(scaler_vector_x437)
    scaler_sharp = preprocessing.StandardScaler().fit(scaler_vector_sharp)
    config={
        "scaler":scaler,
        "clf":clf,
        "pca":'',
        "savepath":'./data/19/',
        "stop":-1,
        "scaler_line":scaler2,
        "clf_line":clf2,       
        "scaler_x437":scaler_x437,
        "clf_x437":clf_x437,        
        "scaler_sharp":scaler_sharp,
        "clf_sharp":clf_sharp,
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
    x437=[]
     
    # mouse=mouses[0]
    # draw_analyst_single(1,mouses[0],0,0,notfind=[1],nmachine=[1])
    # exit()
    # plt.savefig(path+"3d_%3d.png"%idx)
    # plt.clf()
    # plt.close()
    # def getf(mouse):
    #     tmp=[]
    #     # angles=get_distribution_3dangle(mouse)
    #     t=mouse[2]
    #     tmp.append(t[0])
    #     angles=getfivex(t,range(5),5)
    #     for i in range(1,5):
    #         dt=angles[i]-angles[i-1]
    #         tmp.append(dt*10)
    #     return np.array(tmp)

    vtr=[]
    vtrl=[]
    for i in range(n):
        tmp=get_X_PN(mouses[i])
        if mouses[i][0][0]>=437 and tmp==False:
            vtr.append(getf(mouses[i]))
            vtrl.append(labels[i])
    vtr=np.array(vtr)
    vtrl=np.array(vtrl)
    print vtr.shape
    # vtr = preprocessing.scale(vtr)

    # exit()


    # dt=datadeal.DataTrain()
    # clf = MLPClassifier(alpha=0.1,
    #     activation='logistic', \
    #     hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
    #     max_iter=250,early_stopping=True, epsilon=1e-04,\
    #     # learning_rate_init=0.1,learning_rate='invscaling',
    # )
    # # clf=SVC(C=2)
    # np.set_printoptions(formatter={'float':lambda x: "%d"%float(x)})
    # # confusion=dt.trainTest(clf,vector,labels,4.0,classn=6,returnconfusion=True)
    # confusion=dt.trainTest(clf,vtr,vtrl,10.0,classn=2,returnconfusion=True)

    # vtr=[]
    # vtrl=[]
    # for i in range(n):
    #     tmp=get_X_PN(mouses[i])
    #     if mouses[i][0][0]>=437 and i <2600 and tmp==False:
    #         print mouses[i][2][0]
    #         vtr.append(getf(mouses[i]))
    #         vtrl.append(labels[i])

    # for v in vtr:
    #     plt.plot(range(5),v,c='b')
    # plt.show()
    # exit()

    vtr=[]
    for i in range(n):
        tmp=get_X_PN(mouses[i])
        if mouses[i][0][0]>=437 and i in range(0,2600) and tmp==False:
        # if mouses[i][0][0]>=437 and i in range(2600,2900) and tmp==False:
            tmp_rec=getf(mouses[i])
            if tmp_rec[1]>1000:
                continue
            vtr.append(getf(mouses[i]))
    vtr=np.array(vtr)
    # print vtr.shape
    # exit()
    for v in vtr:
        plt.plot(range(3),v,c='g')

    plt.show()
    # print vtr
    # print vtr.shape
    # print x437
    # for i in x437:
    #     draw_analyst_single(i,mouses[i],0,0,notfind=[i],nmachine=[i],\
    #         path='./data/19/notfind2/',save=False)
        # draw_analyst(i,mouses[i],0,0,x437,x437,path='./data/19/notfind2/',stop=2600)
    exit()
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
    print vector.shape
    exit()
    # plt.scatter(range(len(vector)),vector[:,1])
    # plt.show()
    # vector = preprocessing.scale(vector)
    vector2 = preprocessing.scale(vector2)
    # pca = PCA(n_components=9)
    # pca.fit(vector)
    # vector=pca.transform(vector)

    dt=datadeal.DataTrain()
    # clf2 = MLPClassifier(alpha=0.2,
    #     activation='logistic', \
    #     hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
    #     max_iter=250,early_stopping=True, epsilon=1e-04,\
    #     # learning_rate_init=0.1,learning_rate='invscaling',
    # )
    clf=SVC(C=0.2)
    np.set_printoptions(formatter={'float':lambda x: "%d"%float(x)})
    # confusion=dt.trainTest(clf,vector,labels,4.0,classn=6,returnconfusion=True)
    confusion=dt.trainTest(clf2,vector2,labels2,4.0,classn=2,returnconfusion=True)
    # dw=datadraw.DataDraw('2d')
    # confusion=confusion**0.1
    # datadraw.plot_confusion_matrix(confusion,range(6),'a','a',max=confusion.max())

if __name__=="__main__":
    # maintrain()   
    maintest()
