import sys
from pyspark import SparkContext
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn import  metrics

output_path = sys.argv[1]
inputx = sys.argv[2]
# input_all = sys.argv[3]

sc = SparkContext(appName="train")
rdd_train = sc.textFile(inputx)
# rdd_all = sc.textFile(input_all)

# common func =========================
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
    if len(items)==4:
        label=int(items[3])
    else:
        label=0
    return [int(items[0]),np.array([mtr_x,mtr_y,mtr_t]),goal,label]

def get_xpn_mac(mouse):
    x=mouse[0]
    n=len(x)
    for i in range(1,n):
        if x[i-1]>x[i]:
            return False
    return True

def filter_xnp(data):
    pass
    return get_xpn_mac(data[1])

def getfivex(d,idxx):
    n=len(d)
    for i in range(len(idxx)):
        if idxx[i]<0:
            idxx[i]=0
        if idxx[i]>(n-1):
            idxx[i]=n-1
    d=np.array(d)
    return d[idxx].tolist()

# feature ######################################
def get_distx(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])
    x=x/x.max()
    y=y/y.max()
    t=t/t.max()

    angle_dist=np.zeros(5,dtype=np.float)
    sharp=[0.0]
    for i in range(1,n):
        if i+1>=n:
            break
        else:
            vx1=x[i+1]-x[i]
            vy1=y[i+1]-y[i]
            vt1=t[i+1]-t[i]
            vx2=x[i]-x[i-1]
            vy2=y[i]-y[i-1]
            vt2=t[i]-t[i-1]
            dt=t[i+1]-t[i-1]
            angle=vx1*vx2+vy1*vy2
            # angle3d=angle+vt1*vt2
            if dt==0:
                continue
            r13d=(vx1**2+vy1**2+vt1**2)**0.5
            r23d=(vx2**2+vy2**2+vt2**2)**0.5
            if r13d==0 or r23d==0:
                continue
            # angle3d/=r13d
            # angle3d/=r23d
            # arr3d.append(angle3d)
            if angle>-1.1 and angle<=-0.6:
                angle_dist[0]+=1
            elif angle>-0.6 and angle<=-0.3:
                angle_dist[1]+=1
            elif angle>-0.3 and angle<=0.3:
                angle_dist[2]+=1
            elif angle>0.3 and angle<1:
                angle_dist[3]+=1
            elif angle>=1.0:
                angle_dist[4]+=1

            if angle>-1.1 and angle<-0.01:
                rr=r13d if r13d<r23d else r23d
                # sharp.append(angle)
                sharp.append(rr)
    
    n=float(sum(angle_dist))
    feat=[]
    feat.extend(angle_dist.tolist())
    feat.append(float(len(sharp)))
    feat.append(float(sum(sharp))/float(len(sharp)))
    return feat

def get_shape(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])
    x=x/x.max()
    y=y/y.max()
    t=t/t.max()
    
    vxt=[0.0]
    for i in range(0,n):
        if i+1>=n:
            break
        else:
            vx=x[i+1]-x[i]
            dt=t[i+1]-t[i]
            if dt==0:
                continue
            angle=vx/dt
            vxt.append(angle)
    feat=[]

    idx=range(len(vxt)-5,len(vxt))
    lastfive=getfivex(vxt,idx)
    lastfive=np.array(lastfive)
    feat.extend([lastfive.std(),lastfive.mean()])

    idx=range(0,5)
    beginfive=getfivex(vxt,idx)
    beginfive=np.array(beginfive)
    feat.extend([beginfive.std(),beginfive.mean()])

    trate=sum(t[:5])/abs(t[-1])
    feat.append(trate)
 
    idx=range(0,3)
    t3=getfivex(t,idx)
    feat.extend(t3)

    if  len(mouse[2])>1:
        feat.append(1.0 if mouse[2][1]>500 else 0.0)
    else:
        feat.append(0.0)

    # first five angle is concave convex
    xfive=getfivex(x,range(0,5))
    tfive=getfivex(t,range(0,5))
    ax=xfive[4]-xfive[0]
    at=tfive[4]-tfive[0]
    mx=(xfive[4]+xfive[0])/2
    mt=(tfive[4]+tfive[0])/2
    bx=mx-xfive[2]
    bt=mt-tfive[2]
    angle=ax*bx+at*bt
    feat.append(angle)

    return feat

def get_convas(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])
    x=x/x.max()
    y=y/y.max()
    t=t/t.max()
    for i in range(n):
        pass

def get_feats(data):
    mouse=data[1]
    features=[]
    features.extend(get_distx(mouse))
    features.extend(get_shape(mouse))
    data.append(features)
    return data

trains = rdd_train.map(splitx).filter(filter_xnp).map(get_feats)
trains =trains.collect()
vector_shape=[]
lables_shape=[]
for d in trains:
    idx=d[0]
    label=d[3]
    f1=d[4]
    # f2=d[5]
    # f1.extend(f2)
    vector_shape.append(f1)
    lables_shape.append(label)

vector_shape=np.array(vector_shape) 
np.set_printoptions(formatter={'float':lambda x: "%5.2f"%float(x)})   
print vector_shape[:5]
print "===="
print vector_shape[len(vector_shape)-5:len(vector_shape)]
lables_shape=np.array(lables_shape) 
scaler_shape = preprocessing.StandardScaler().fit(vector_shape)
vector_shape = preprocessing.scale(vector_shape)


def testDs(data,conf):
    scalar=conf["scalar"]
    clf=conf["clf"]    
    f1=data[4]
    # f2=data[5]
    # f1.extend(f2)
    f1=scalar.transform([f1])
    r=clf.predict(f1)
    if r[0]==0: # machine
        return True
    else:
        return False

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

clf=SVC(C=1.5)

trainTest(clf,vector_shape,lables_shape)