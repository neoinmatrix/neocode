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

def get_sharp_angle(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    xn=len(mouse[0])
    angle_arr=[]
    r_arr=[]
    # aspeed_arr=[0.0]
    for i in range(1,xn):
        if i+1>=xn:
            break
        else:
            vx1=x[i+1]-x[i]
            vy1=y[i+1]-y[i]
            vx2=x[i]-x[i-1]
            vy2=y[i]-y[i-1]
            dt=t[i+1]-t[i-1]
            angle=(vx1*vx2+vy1*vy2)
            if vx1==0 and vy1==0:
                continue
            if vx2==0 and vy2==0:
                continue
            if dt==0:
                continue
            r1=(vx1**2+vy1**2)**0.5
            r2=(vx2**2+vy2**2)**0.5
            angle/=r1
            angle/=r2
            if angle>-1.0 and angle<-0.0:
                # print angle
                rr=r1 if r1<r2 else r2
                angle_arr.append(angle)
                r_arr.append(rr)
                # print rr
    r_arr=np.array(r_arr)
    if len(r_arr)>1 and r_arr.mean()>40.0:
        return True
    return False
    # angle_arr=np.array(angle_arr)
    # r_arr=np.array(r_arr)
    # if len(r_arr)==0:
    #     return [float(len(r_arr)),0.0,float(len(r_arr))]
    # return [float(len(r_arr)),r_arr.mean(),float(len(r_arr))]
    # tmp=[len(r_arr)]
    # tmp.extend(getMMMS(angle_arr))
    # tmp.extend(getMMMS(r_arr))

def get_distribution_angle(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    xn=len(mouse[0])
    one_arr=[]
    rt_arr=[]

    r_arr=[]
    # aspeed_arr=[0.0]
    for i in range(1,xn):
        if i+1>=xn:
            break
        else:
            vx1=x[i+1]-x[i]
            vy1=y[i+1]-y[i]
            vx2=x[i]-x[i-1]
            vy2=y[i]-y[i-1]
            dt=t[i+1]-t[i-1]
            angle=(vx1*vx2+vy1*vy2)
            if vx1==0 and vy1==0:
                continue
            if vx2==0 and vy2==0:
                continue
            if dt==0:
                continue
            r1=(vx1**2+vy1**2)**0.5
            r2=(vx2**2+vy2**2)**0.5
            angle/=r1
            angle/=r2
            if angle>0 and angle<1:
                rt_arr.append(angle)
            if angle>0.9 and angle<1.1:
                one_arr.append(angle)
    return [float(xn),float(len(rt_arr)),float(len(one_arr))]
 
def get_fft_angle(mouse):
    x=mouse[0]
    y=mouse[1]
    # t=mouse[2]
    xn=len(mouse[0])
    xf=x[xn-1]-x[0]
    yf=y[xn-1]-y[0]
    xf= 1 if xf==0 else 1000.0/xf
    yf= 1 if yf==0 else 100.0/yf
    x=x*xf
    y=y*yf

    angle_arr=[]
    r_arr=[]
    # aspeed_arr=[0.0]
    for i in range(1,xn):
        if i+1>=xn:
            break
        else:
            vx1=x[i+1]-x[i]
            vy1=y[i+1]-y[i]
            vx2=x[i]-x[i-1]
            vy2=y[i]-y[i-1]
            dt=t[i+1]-t[i-1]
            angle=(vx1*vx2+vy1*vy2)
            if vx1==0 and vy1==0:
                continue
            if vx2==0 and vy2==0:
                continue
            if dt==0:
                continue
            r1=(vx1**2+vy1**2)**0.5
            r2=(vx2**2+vy2**2)**0.5
            angle/=r1
            angle/=r2
            if angle>-1.0 and angle<-0.0:
                # print angle
                rr=r1 if r1<r2 else r2
                angle_arr.append(angle)
                r_arr.append(rr)
                # print rr
    r_arr=np.array(r_arr)
    if len(r_arr)>1 and r_arr.mean()>40.0:
        return True
    return False

def getfeature(idx,mouse,goal,label,use_all=False):
    tmp=[]
    if use_all==True:
        pass
    else:
        # has changed x towards so must not be machine
        if get_X_PN(mouse)==True:  
            return False
        # the x >437 must be machine 
        if mouse[0][0]>=437:
            return True
        
        sangle=get_sharp_angle(mouse)
        if sangle==True:
            return True
        anglenum=get_distribution_angle(mouse)
        # if int(idx)==139:
        #     print anglenum
        #     exit()
        if anglenum[2]/anglenum[0]>0.5:
            if anglenum[1]>0.0 and anglenum[2]/anglenum[1]>3:
                return True
    # anglenum=get_distribution_angle(mouse)
    tmp.extend([0.0])
    return np.array(tmp).reshape([1,len(tmp)])

def testResultAll(clf,scaler,pca,savepath='./data/0711tmp.txt',stop=1200):
    ds=dataset.DataSet()
    allnum=0
    mclass=[[],[],[]]
    while True:
        idx,mouse,goal,label=ds.readTestFile()
        if idx==False:
            break
        tmp=getfeature(idx,mouse,goal,label)
        if type(tmp) is bool:
            if tmp==False:
                pass
            elif tmp==True:
                mclass[0].append(idx)
                mclass[1].append(idx)
        else:
            pass
            # tmp=scaler.transform(tmp)
            # tmp=pca.transform(tmp)
            # r=clf.predict(tmp)
            # if r[0]>0:
            #     pass
            #     # print idx
            #     if int(idx)>1000:
            #         exit()
            #     if int(idx) in [139,249,253,359,370,416,435,478,483,485,505,516,542,548,549,630,691,703,721,740,777,781,786,867,873,987,1002]:
            #         print tmp
            #         # print idx,len(mouse[0])
            #         # fig = plt.figure()  
            #     #     ax = fig.add_subplot(111, projection='3d') 
            #     #     ax.plot(mouse[0],mouse[1],mouse[2])
            #     #     path='./data/notfind/'
            #     #     plt.savefig(path+"3d%s.png"%idx)
            #     #     plt.clf()
            #     #     plt.close()
            #     #     # plt.show()
            #     #     # exit()
            #     #     pass
            # else:
            #     # pass
            #     mclass[0].append(idx)
            #     mclass[2].append(idx)

        if stop!=-1 and allnum>stop:
            break
        if allnum%1000==0:
            print idx,len(mclass[0]),len(mclass[1]),len(mclass[2])
        allnum+=1
    print "all:",len(mclass[0]),len(mclass[1]),len(mclass[2])
    savestr=createstr(mclass[0])
    with open(savepath,'w') as f:
        f.write(savestr)
    print "ok"

def plot_correlation(cm, genre_list, name, title):
    from matplotlib import pylab 
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Greens', vmin=-1.0, vmax=1.0)
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

def analyst_correclation():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    vector_label=[]
    for i in range(n):
        tmp=getfeature(0,mouses[i],goals[i],0,is_train=True)
        # tmp=getfeature(0,mouses[i],goals[i],0,is_train=False)
        if type(tmp) is bool:
            continue  
        vector.append(tmp[0])
        vector_label.append(labels[i])

    vector=np.array(vector)
    labels=np.array(vector_label)
    scaler_vector=vector
    vector = preprocessing.scale(vector)
    vector=np.c_[vector,labels]
    # print vector.shape
    cov=np.cov(vector.T)
    cov=np.abs(cov)
    print cov[66]
    plot_correlation(cov, range(len(cov[0])), 'confusion_matrix', 'confusion_matrix')
    exit()

def main():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    labels_tmp=[]
    # print get_distribution_angle(mouses[0])
    # exit()
    # print mouses[81][0]
    # print mouses[81][1]
    # exit()
    for i in range(n):
        # tmp=getfeature(i,mouses[i],goals[i],0,use_all=False)
        if i in range(2650,2700):
            tmp=getfeature(0,mouses[i],goals[i],0,use_all=True)
        else:
            continue
            # tmp=getfeature(0,mouses[i],goals[i],0,use_all=False)
        if type(tmp) is bool:
            continue  
        vector.append(tmp[0])
        labels_tmp.append(labels[i])
    labels=np.array(labels_tmp)
    vector=np.array(vector)
    scaler_vector=vector
    # print vector
    # exit()
    # plt.scatter(range(len(vector)),vector[:,1])
    # plt.show()
    # exit()
    # print vector.shape
    # print vector[0]
    # print vector[1000]
    # exit()
    # vector = preprocessing.scale(vector)
    pca = PCA(n_components=9)
    # pca.fit(vector)
    # vector=pca.transform(vector)

    dt=datadeal.DataTrain()
    # about 17 w
    clf = MLPClassifier(alpha=0,
        activation='logistic', \
        hidden_layer_sizes=(12,15),random_state=0,solver='lbfgs',\
        max_iter=650,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )
    # clf=SVC(C=3)
    # print vector.shape
    # exit()
    # # False
    test=False
    if test==True:
        np.set_printoptions(formatter={'float':lambda x: "%d"%float(x)})
        dt.trainTest(clf,vector,labels,4.0)
    else:
        clf.fit(vector,labels)
        scaler = preprocessing.StandardScaler().fit(scaler_vector)
        testResultAll(clf,scaler,pca,savepath='./data/0712tmp.txt',stop=-1)

def mainold():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]

    # get_sharp_angle(mouses[2921])
    # print "==============="
    # get_sharp_angle(mouses[884])
    # exit()
    vector=[]
    vector_label=[]
    labels_tmp=labels.tolist()
    goals_tmp=goals.tolist()
    for i in range(2200):
        idx=np.random.randint(2600,3000)
        tmp=mouses[idx]
        mouses.append(tmp)
        labels_tmp.append(labels[idx])
        goals_tmp.append(goals[idx])
    labels=np.array(labels_tmp)
    goals=np.array(goals_tmp)

    for i in range(len(labels)):
        tmp=getfeature(0,mouses[i],goals[i],0,is_train=False)
        # tmp=getfeature(0,mouses[i],goals[i],0,is_train=False)
        if type(tmp) is bool:
            continue  
        vector.append(tmp[0])
        vector_label.append(labels[i])
    # for i in range(2200):
    #     idx=np.random.randint(2600,3000)
    #     tmp=vector[idx]
    #     vector.append(tmp)
    #     vector_label.append(labels[idx])

    vector=np.array(vector)
    labels=np.array(vector_label)
    scaler_vector=vector
    # print vector.shape
    # exit()
    vector = preprocessing.scale(vector)
    pca = PCA(n_components=30)
    pca.fit(vector)
    vector=pca.transform(vector)

    dt=datadeal.DataTrain()
    # about 17 w
    clf = MLPClassifier(alpha=0,
        activation='logistic', \
        hidden_layer_sizes=(30,30),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )
    # clf=SVC(C=3)

    # # False
    test=False
    if test==True:
        np.set_printoptions(formatter={'float':lambda x: "%d"%float(x)})
        dt.trainTest(clf,vector,labels,4.0)
    else:
        clf.fit(vector,labels)
        scaler = preprocessing.StandardScaler().fit(scaler_vector)
        testResultAll(clf,scaler,pca,savepath='./data/0712tmp.txt',stop=12000)

if __name__=="__main__":
    main()
    pass