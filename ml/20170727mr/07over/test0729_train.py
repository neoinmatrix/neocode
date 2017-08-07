import sys
from pyspark import SparkContext
import numpy as np
# from sklearn.model_selection import KFold
from sklearn.cross_validation import KFold
from sklearn.cross_validation import ShuffleSplit
# from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import  metrics
# import matplotlib.pyplot as plt 
from sklearn import preprocessing
# from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

output = sys.argv[1]
input = sys.argv[2]
sc = SparkContext(appName="train")
rdd_train = sc.textFile(input)

#  data filter =========================
def get_X_PN_isMachine(mouse):
    x=mouse[0]
    n=len(x)
    for i in range(1,n):
        if x[i-1]>x[i]:
            return False
    return True

def get_437(mouse):
    x=mouse[0]
    if x[0]<437:
        return False
    else:
        return True

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
    mean=sum(r_arr)/float(len(r_arr)) if len(r_arr)>0 else 0
    if len(r_arr)>1 and mean>40.0:
        return True
    else:
        return False

def filterx(data):
    mouse=data[1]
    flag=True
    if get_X_PN_isMachine(mouse)==False:
        flag=False
    else:
        if get_437(mouse)==False:
            flag=False
        if flag==False and get_sharp_angle(mouse)==True:
            flag=True
    return [data[0],flag]

#  data format =========================
def splitx(raw):
    items=raw.split(' ')
    mtr_x=[]
    mtr_y=[]
    mtr_t=[]
    for v in items[1].split(';'):
        tmp=v.split(',')
        if len(tmp)!=3:
            continue
        mtr_x.append(float(tmp[0]))
        mtr_y.append(float(tmp[1]))
        mtr_t.append(float(tmp[2]))
    gtmp=items[2].split(',')
    goal=[float(gtmp[0]),float(gtmp[1])]
    label=int(items[3])
    return [items[0],[mtr_x,mtr_y,mtr_t],goal,label]

def filterXNP(data):
    pass
    return get_X_PN_isMachine(data[1])

def getfivex(d,idxx,num=5):
    xn=len(d)
    for i in range(num):
        if idxx[i]<0:
            idxx[i]=0
        if idxx[i]>(xn-1):
            idxx[i]=xn-1
    d=np.array(d)
    return d[idxx].tolist()

def getf(mouse):
    tmp=[]
    t=mouse[2]
    tmp.append(t[0])
    angles=getfivex(t,range(5),5)
    angles=np.array(angles)
    tmp.append(angles.std())
    tmp.append(angles.mean())
    return tmp

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
    return tmp

def getFeatures(data):
    mouse=data[1]
    features=[]
    features.extend(getf(mouse))
    features.extend(getf(mouse))
    data[1]=features
    return data

trains = rdd_train.map(splitx).filter(filterXNP).map(getFeatures)
trains =trains.collect()
vector=[]
lables=[]
for d in trains:
    vector.append(d[1])
    lables.append(d[3])
vector=np.array(vector)    
lables=np.array(lables)    
vector = preprocessing.scale(vector)

def trainTest(clf,X,y,fold=10.0,classn=2,returnconfusion=False):
    # kf = KFold(n_splits=int(fold), shuffle=True,random_state=np.random.randint(len(y)))
    # kf = KFold(len(y),n_folds=int(fold))
    kf = ShuffleSplit(len(y), n_iter=int(fold), test_size=0.25, random_state=0)
    accuracy=0.0
    confusion=np.zeros([classn,classn])
    for train_index, test_index in kf:
        X_train=X[train_index]
        y_train=y[train_index]
        X_test=X[test_index]
        expected=y[test_index]
        clf.fit(X_train,y_train)
        predicted = clf.predict(X_test)
        accy_tmp=metrics.accuracy_score(expected, predicted)
        accuracy+=accy_tmp
        conf_tmp=metrics.confusion_matrix(expected, predicted)
        confusion+=conf_tmp
        print "predited rate:%f"%accy_tmp
    print confusion
    print accuracy/fold
    if returnconfusion:
        return confusion

# clf=SVC(C=0.5)
clf=RandomForestClassifier(n_estimators=10)
# print len(trains)
# print trains[3]
trainTest(clf,vector,lables)

# result.saveAsTextFile(output)