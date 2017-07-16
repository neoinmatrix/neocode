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

def get_distribution_2dangle(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    x=x/x.max()
    y=y/y.max()
    t=t/t.max()
    xn=len(mouse[0])
    one_arr=[]
    rt_arr=[]

    r_arr=[]
    angle_arr=np.zeros(5,dtype=np.float)
    angle_arr3d=np.zeros(5,dtype=np.float)
    angle_xrr=np.zeros(5,dtype=np.float)
    angle_yrr=np.zeros(5,dtype=np.float)
    # aspeed_arr=[0.0]
    arr=[]
    arr3d=[]
    for i in range(1,xn):
        if i+1>=xn:
            break
        else:
            vx1=x[i+1]-x[i]
            vy1=y[i+1]-y[i]
            vt1=t[i+1]-t[i]
            vx2=x[i]-x[i-1]
            vy2=y[i]-y[i-1]
            vt2=t[i]-t[i-1]
            dt=t[i+1]-t[i-1]
            angle=(vx1*vx2+vy1*vy2)
            angle3d=(angle+vt1*vt2)
            if dt==0:
                continue
            r1=(vx1**2+vy1**2)**0.5
            r2=(vx2**2+vy2**2)**0.5
            r13d=(vx1**2+vy1**2+vt1**2)**0.5
            r23d=(vx2**2+vy2**2+vt2**2)**0.5

            if r1==0 or r2==0 or r13d==0 or r23d==0:
                continue
            angle/=r1
            angle/=r2
            angle3d/=r13d
            angle3d/=r23d
            arr.append(angle)
            arr3d.append(angle3d)
            if angle>-1.1 and angle<=-0.6:
                angle_arr[0]+=1
                angle_arr3d[0]+=1
                angle_xrr[0]+=vx1
                angle_yrr[0]+=vy1
            elif angle>-0.6 and angle<=-0.3:
                angle_arr[1]+=1
                angle_arr3d[1]+=1
                angle_xrr[1]+=vx1
                angle_yrr[1]+=vy1
            elif angle>-0.3 and angle<=0.3:
                angle_arr[2]+=1
                angle_arr3d[2]+=1
                angle_xrr[2]+=vx1
                angle_yrr[2]+=vy1
            elif angle>0.3 and angle<1:
                angle_arr[3]+=1
                angle_arr3d[3]+=1
                angle_xrr[3]+=vx1
                angle_yrr[3]+=vy1
            elif angle>=1.0:
                angle_arr[4]+=1
                angle_arr3d[4]+=1
                angle_xrr[4]+=vx1
                angle_yrr[4]+=vy1
    
    n=float(sum(angle_arr))
    # angle_arr=angle_arr/float(n)
    tmp=[]
    # tmp.extend(angle_arr.tolist())
    tmp.extend(angle_arr3d.tolist())
    # angle_arr.extend(angle_xrr)
    # angle_arr.extend(angle_yrr)
    # angle_arr=angle_arr3d.tolist()
    tmp.append(n)
    lastfive=getfivex(arr3d,range(len(arr3d)-5,len(arr3d)))

    tmp.extend(lastfive)
    # tmp.append(x[0])
    # tmp.append(y[0])
    # tmp.append(t[0])
    return tmp
    # return arr
 
def get_distribution_3dangle(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    x=x/x.max()
    y=y/y.max()
    t=t/t.max()
    xn=len(mouse[0])

    angle_arr3d=np.zeros(5,dtype=np.float)
    arr3d=[0.0]
    for i in range(1,xn):
        if i+1>=xn:
            break
        else:
            vx1=x[i+1]-x[i]
            vy1=y[i+1]-y[i]
            vt1=t[i+1]-t[i]
            vx2=x[i]-x[i-1]
            vy2=y[i]-y[i-1]
            vt2=t[i]-t[i-1]
            dt=t[i+1]-t[i-1]
            angle=(vx1*vx2+vy1*vy2)
            angle3d=(angle+vt1*vt2)
            if dt==0:
                continue
            r13d=(vx1**2+vy1**2+vt1**2)**0.5
            r23d=(vx2**2+vy2**2+vt2**2)**0.5
            if r13d==0 or r23d==0:
                continue
            angle3d/=r13d
            angle3d/=r23d
            arr3d.append(angle3d)
            if angle>-1.1 and angle<=-0.6:
                angle_arr3d[0]+=1
            elif angle>-0.6 and angle<=-0.3:
                angle_arr3d[1]+=1
            elif angle>-0.3 and angle<=0.3:
                angle_arr3d[2]+=1
            elif angle>0.3 and angle<1:
                angle_arr3d[3]+=1
            elif angle>=1.0:
                angle_arr3d[4]+=1
    
    n=float(sum(angle_arr3d))
    # angle_arr=angle_arr/float(n)
    tmp=[]
    # tmp.extend(angle_arr.tolist())
    tmp.extend(angle_arr3d.tolist())
    # tmp.append(n)
    # lastfive=getfivex(arr3d,range(len(arr3d)-5,len(arr3d)))
    # tmp.extend(lastfive)
    # tmp.append(mouse[2][0])
    # tmp.append(mouse[2][-1])
    return tmp

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
        # return 1
        # anglenum=get_distribution_angle(mouse)
        # if int(idx)==139:
        #     print anglenum
        #     exit()
        # if anglenum[2]/anglenum[0]>0.8:
        #     if anglenum[1]>0.0 and anglenum[1]/anglenum[0]<0.05:
        #         return True
    # anglenum=get_distribution_3dangle_rect_five(mouse)
    # anglenum=get_distribution_3dangle_five(mouse)
    anglenum=get_distribution_3dangle(mouse)
    tmp.extend(anglenum)
    # dv=get_derivative(mouse)
    # tmp.extend(dv)
    return np.array(tmp)

def testResultAll(clf,scaler,pca,savepath='./data/0713tmp.txt',stop=1200):
    ds=dataset.DataSet()
    allnum=0
    mclass=[[],[],[],[],[],[],[],[]]
    rclass=[[],[],[],[],[],[],[],[]]
    np.set_printoptions(formatter={'float':lambda x: "%5.5f"%float(x)})
    inarea=[0]*3

    # fig = plt.figure()  
    # ax = fig.add_subplot(111, projection='3d') 
    # ax.plot(mouse[0],mouse[1],mouse[2])
    while True:
        idx,mouse,goal,label=ds.readTestFile()
        if idx==False:
            break
        # if idx==136:
        #     print getfeature(idx,mouse,goal,label)
        # if idx==139:
        #     # print mouse

        #     print getfeature(idx,mouse,goal,label)
        #     ax=plt.figure()
        #     fig = plt.figure()  
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter(mouse[0],mouse[1],mouse[2])
        #     plt.show()
        #     exit()
        # if idx in [136,139,143,145]:
        #     # ax=plt.figure()
        #     # plt.plot(mouse[0],mouse[1])
        #     # plt.savefig('./data/notfind/3dx%d.png'%idx)
        #     fig = plt.figure()  
        #     ax = fig.add_subplot(111, projection='3d')
        #     if idx in [139]:
        #         c='b'
        #     else:
        #         c='g' 
        #     ax.plot(mouse[0],mouse[1],mouse[2],c=c)

        #     plt.title(str(idx))
        #     plt.savefig("./data/notfind/3dx%d.png"%idx)
        #     plt.clf()
        #     plt.close()
        # if idx in [136,139,143,145]:
        #     # ax=plt.figure()
        #     # plt.plot(mouse[0],mouse[1])
        #     # plt.savefig('./data/notfind/3dx%d.png'%idx)
        #     if idx in [139]:
        #         c='b'
        #     else:
        #         c='g' 

        #     plt.plot(mouse[0],mouse[1],c=c)
        #     plt.title(str(idx))
        #     plt.savefig("./data/notfind/2dy%d.png"%idx)
        #     plt.clf()
        #     plt.close()

        # if idx>200:
        #     exit()
        tmp=getfeature(idx,mouse,goal,label)
        if type(tmp) is bool:
            if tmp==False:
                pass
            elif tmp==True:
                mclass[0].append(idx)
                mclass[1].append(idx)
        else:
            tmp=scaler.transform([tmp])
            # tmp=pca.transform(tmp)
            r=clf.predict(tmp)
            rclass[0].append(idx)
            if r[0]==1: # >0 is manual
                rclass[1].append(idx)
            elif r[0]==2:
                rclass[2].append(idx)
            elif r[0]==3:
                rclass[3].append(idx)
            elif r[0]==4:
                rclass[4].append(idx)
            else:
                rclass[5].append(idx)

        if stop!=-1 and allnum>stop:
            break
        if allnum%1000==0:
            print idx,len(mclass[0]),len(mclass[1])
            print len(rclass[0]),len(rclass[1]),len(rclass[2]),len(rclass[3]),len(rclass[4]),len(rclass[5])
            print "======"
        allnum+=1
    print "all:",len(mclass[0]),len(mclass[1]),len(rclass[0])
    savestr=createstr(mclass[0])
    with open(savepath,'w') as f:
        f.write(savestr)
   
    for i in range(1,6):
        savestr=createstr(rclass[i])
        with open("./data/15/r_%d.txt"%i,'w') as f:
            f.write(savestr)
    # savestr=createstr(mclass[2])
    # with open("./data/0714_class.txt",'w') as f:
    #     f.write(savestr)    
    # savestr=createstr(mclass[3])
    # with open("./data/0714_class0.txt",'w') as f:
    #     f.write(savestr)

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
    idxs=np.random.randint(0,2600,50)
    # print get_distribution_angle(mouses[0])
    # exit()
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

    labels=np.array(labels_tmp)
    vector=np.array(mouses_tmp)
    scaler_vector=vector
    np.set_printoptions(formatter={'float':lambda x: "%5.3f"%float(x)})
    # print vector.shape
    # print vector
    # exit()
    # plt.scatter(range(len(vector)),vector[:,3])
    # plt.show()
    # exit()
    vector = preprocessing.scale(vector)
    pca = PCA(n_components=9)
    # pca.fit(vector)
    # vector=pca.transform(vector)

    dt=datadeal.DataTrain()
    # about 17 w
    clf = MLPClassifier(alpha=0.5,
        activation='logistic', \
        hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
        # hidden_layer_sizes=(15,15),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )
    clf.fit(vector,labels)
    scaler = preprocessing.StandardScaler().fit(scaler_vector)
    testResultAll(clf,scaler,pca,savepath='./data/15/07tmp.txt',stop=-1)

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
    for i in range(n):
        tmp=getfeature(i,mouses[i],goals[i],0,use_all=True)
        # tmp=getfeaturetest(i,mouses[i],goals[i],0,use_all=False)
        if type(tmp) is bool:
            # print i,tmp
            continue
        if i in range(0,2600):
            labels_tmp.append(0)
        elif i in range(2600,2650):
            # labels_tmp.append(1)
            labels_tmp.append(1)
        elif i in range(2650,2700):
            # labels_tmp.append(2)
            labels_tmp.append(2)
            # labels_tmp.append(2)
        elif i in range(2700,2800):
            labels_tmp.append(3)
        elif i in range(2800,2900):
            labels_tmp.append(4)
        else:
            labels_tmp.append(5)
        mouses_tmp.append(tmp)
        # if i in range(2600,2700):
        #     mouses_tmp.append(tmp)
        # if i in range(2650,2700):
        #     mouses_tmp.append(tmp)
        # labels_tmp.append(labels[i])

    labels=np.array(labels_tmp)
    # print len(mouses_tmp)
    vector=np.array(mouses_tmp)
    scaler_vector=vector
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
    vector = preprocessing.scale(vector)
    # pca = PCA(n_components=9)
    # pca.fit(vector)
    # vector=pca.transform(vector)

    dt=datadeal.DataTrain()
    # about 17 w
    clf = MLPClassifier(alpha=0.5,
        activation='logistic', \
        # 11,11 
        hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )
    np.set_printoptions(formatter={'float':lambda x: "%d"%float(x)})
    confusion=dt.trainTest(clf,vector,labels,4.0,classn=6,returnconfusion=True)
    # dw=datadraw.DataDraw('2d')
    # genre_list, name, title,max,save=False
    confusion=confusion**0.1
    # datadraw.plot_confusion_matrix(confusion,range(6),'a','a',max=confusion.max())

def main():
    # maintrain()   
    maintest()

if __name__=="__main__":
    main()
    pass
