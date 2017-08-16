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

def get_t_T(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    x=x/x.max()
    y=y/y.max()
    t=t/t.max()
    xn=len(mouse[0])

    angle_arr3d=np.zeros(5,dtype=np.float)
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
    # angle_arr=angle_arr/float(n)
    tmp=[]
    # tmp.extend(angle_arr.tolist())
    # tmp.extend(angle_arr3d.tolist())
    # tmp.append(n)

    lastfive=getfivex(arr3d,range(len(arr3d)-5,len(arr3d)))
    tmp.extend(lastfive)
    beginfive=getfivex(arr3d,range(0,5))
    tmp.extend(beginfive)
    trate=sum(t[:5])/abs(t[-1])
    tmp.append(trate)

    tfive=getfivex(t,range(0,5))
    tmp.extend(tfive[:3])
    if  len(mouse[2])>1:
        tmp.append(1.0 if mouse[2][1]>500 else 0.0)
        # tmp.append(1.0 if mouse[2][1]>500 else 0.0)
    else:
        tmp.append(0.0)
        # tmp.append(0.0)
    # tmp.append(tfive[4])
    # first five ===================
    xfive=getfivex(x,range(0,5))
    ax=xfive[4]-xfive[0]
    at=xfive[4]-tfive[0]
    mx=(xfive[4]+xfive[0])/2
    mt=(tfive[4]+tfive[0])/2
    bx=mx-xfive[2]
    bt=mt-tfive[2]
    angle=ax*bx+at*bt
    tmp.append(angle)
    # tmp.append(angle)


    # first three ===================
    # xfive=getfivex(x,range(0,3),3)
    # ax=xfive[2]-xfive[0]
    # at=xfive[2]-tfive[0]
    # mx=(xfive[2]+xfive[0])/2
    # mt=(tfive[2]+tfive[0])/2
    # bx=mx-xfive[1]
    # bt=mt-tfive[1]
    # angle=ax*bx+at*bt
    # tmp.append(angle)
    # tratel=sum(t[-5:])/abs(t[-1])
    # tmp.append(tratel)
    # tmp.append(abs(mouse[2][0])**0.4)
    # tmp.append(abs(mouse[2][-1])**0.4)
    # tmp.append((mouse[2][0])**0.4)
    # tmp.append((mouse[2][-1])**0.4)
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
        # exists acute angle must be machine 
        sangle=get_sharp_angle(mouse)
        if sangle==True:
            return True
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
    notfind=[1,5,45,50,55,57,58,62,63,65,98,99,100,111,132,145,170,182,183,202,226,232,260,266,304,353,356,362,365,371,373,380,390,394,396,423,457,468,477,484,490,504,506,509,534,548,606,630,654,659,671,680,691,693,695,701,705,717,738,743,748,751,753,757,760,772,781,790,801,816,832,836,842,870,879,886,888,907,920,921,941,968,974,989,990,993,997]
    nmachine=[548,630,691,781]
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

    if idx>1000:
        exit()

def draw_analyst_single(idx,mouse,goal,label):
    pass
    # notfind=[1,5,45,50,55,57,58,62,63,65,98,99,100,111,132,145,170,182,183,202,226,232,260,266,304,353,356,362,365,371,373,380,390,394,396,423,457,468,477,484,490,504,506,509,534,548,606,630,654,659,671,680,691,693,695,701,705,717,738,743,748,751,753,757,760,772,781,790,801,816,832,836,842,870,879,886,888,907,920,921,941,968,974,989,990,993,997]
    notfind=[1,5,45,50,548,630,691,781]
    nmachine=[548,630,691,781]
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

    if idx>1000:
        exit()


def testResultAll(clf,scaler,pca,savepath='./data/0717tmp.txt',stop=1200,clf_extra='',scaler_extra=''):
    ds=dataset.DataSet()
    allnum=0
    mclass=[[],[],[],[],[],[],[],[]]
    rclass=[[],[],[],[],[],[],[],[]]
    np.set_printoptions(formatter={'float':lambda x: "%5.5f"%float(x)})
    while True:
        allnum+=1
        idx,mouse,goal,label=ds.readTestFile()
        if idx==False:
            break
        # draw_analyst_single(idx,mouse,goal,label)
        # draw_analyst(idx,mouse,goal,label)
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
            if r[0]==1:
                # this class is other line type 
                tmp=getfeature2(idx,mouse,goal,label)
                tmp=scaler_extra.transform([tmp])
                rr=clf_extra.predict(tmp)
                if rr[0]==0:
                    mclass[0].append(idx)
                    rclass[1].append(idx)
                pass
            else:
                # this class is L or \ line type
                rclass[2].append(idx)
                tmp=getfeature2(idx,mouse,goal,label)
                tmp=scaler_extra.transform([tmp])
                rr=clf_extra.predict(tmp)
                if rr[0]==0:
                    # this is according dx/dt find machine
                    # mclass[0].append(idx)
                    mclass[0].append(idx)
                    rclass[3].append(idx)
                else:
                    # this is not machine
                    rclass[4].append(idx)

        if stop!=-1 and allnum>stop:
            break
        if allnum%1000==0:
            tmp1=''
            for i in range(len(mclass)):
                if len(mclass[i])>0:
                    tmp1+="%d "%len(mclass[i])
            print idx,tmp1
            tmp2=''
            for i in range(len(rclass)):
                if len(rclass[i])>0:
                    tmp2+="%d "%len(rclass[i])
            print tmp2
            print "======"
      
    # ===print=========================================  
    tmp1=''
    print "summary:"
    for i in range(len(mclass)):
        if len(mclass[i])>0:
            tmp1+="%d "%len(mclass[i])
    print idx,tmp1

    tmp2=''
    for i in range(len(rclass)):
        if len(rclass[i])>0:
            tmp2+="%d "%len(rclass[i])
    print tmp2

    # ===save=========================================
    for i in range(len(mclass)):
        if len(mclass[i])==0:
            continue
        savestr=createstr(mclass[i])
        with open("./data/17/m_%d.txt"%i,'w') as f:
            f.write(savestr)
        
    for i in range(len(rclass)):
        if len(rclass[i])==0:
            continue
        savestr=createstr(rclass[i])
        with open("./data/17/r_%d.txt"%i,'w') as f:
            f.write(savestr)

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
    # idxs=np.random.randint(0,2600,50)
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

    for i in range(n):
        tmp=getfeature2(i,mouses[i],goals[i],0,use_all=False)
        # tmp=getfeaturetest(i,mouses[i],goals[i],0,use_all=False)
        if type(tmp) is bool:
            # print i,tmp
            continue
        labels_tmp2.append(labels[i])
        mouses_tmp2.append(tmp)


    labels2=np.array(labels_tmp2)
    vector2=np.array(mouses_tmp2)
    # for i in range(20):
    #     plt.plot(range(7),vector2[i],c='g')
    # for i in range(1300,1320):
    #     plt.plot(range(7),vector2[i],c='b')
    # plt.show()
    # print vector2.shape
    # exit()
    labels=np.array(labels_tmp)
    vector=np.array(mouses_tmp)
    scaler_vector=vector
    scaler_vector2=vector2
    np.set_printoptions(formatter={'float':lambda x: "%5.3f"%float(x)})
    # print vector2.shape
    # print vector
    # exit()
    # plt.scatter(range(len(vector)),vector[:,3])
    # plt.show()
    # exit()
    vector = preprocessing.scale(vector)
    vector2 = preprocessing.scale(vector2)
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
    clf2 = MLPClassifier(alpha=0.2,
        activation='logistic', \
        # hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
        hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
        # hidden_layer_sizes=(15,15),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )
    clf.fit(vector,labels)
    clf2.fit(vector2,labels2)
    scaler = preprocessing.StandardScaler().fit(scaler_vector)
    scaler2 = preprocessing.StandardScaler().fit(scaler_vector2)
    testResultAll(clf,scaler,pca,savepath='./data/17/0717tmp.txt',stop=-1,clf_extra=clf2,scaler_extra=scaler2)

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
        # tmp=getfeaturetest(i,mouses[i],goals[i],0,use_all=False)
        if type(tmp) is bool:
            # print i,tmp
            continue
        labels_tmp2.append(labels[i])
        mouses_tmp2.append(tmp)


    labels2=np.array(labels_tmp2)
    vector2=np.array(mouses_tmp2)
    # print vector2
    # exit(0)

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
    # vector = preprocessing.scale(vector)
    vector2 = preprocessing.scale(vector2)
    # print vector2.shape
    # exit()
    # pca = PCA(n_components=9)
    # pca.fit(vector)
    # vector=pca.transform(vector)

    dt=datadeal.DataTrain()
    # about 17 w
    clf2 = MLPClassifier(alpha=0.2,
        activation='logistic', \
        # 11,11 
        hidden_layer_sizes=(11,11),random_state=0,solver='lbfgs',\
        max_iter=250,early_stopping=True, epsilon=1e-04,\
        # learning_rate_init=0.1,learning_rate='invscaling',
    )
    np.set_printoptions(formatter={'float':lambda x: "%d"%float(x)})
    # confusion=dt.trainTest(clf,vector,labels,4.0,classn=6,returnconfusion=True)
    confusion=dt.trainTest(clf2,vector2,labels2,4.0,classn=2,returnconfusion=True)
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
