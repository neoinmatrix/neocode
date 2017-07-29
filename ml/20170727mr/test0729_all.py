import sys
from pyspark import SparkContext
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing

output_path = sys.argv[1]
inputx = sys.argv[2]
input_all = sys.argv[3]

sc = SparkContext(appName="train")
rdd_train = sc.textFile(inputx)
rdd_all = sc.textFile(input_all)

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
    if len(items)==4:
        label=int(items[3])
    else:
        label=0
    return [int(items[0]),np.array([mtr_x,mtr_y,mtr_t]),goal,label]

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

# shape ######################################
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
    tmp=[]
    tmp.extend(angle_arr3d.tolist())
    return tmp

#f1
def getFs_shape(data):
    mouse=data[1]
    features=[]
    features.extend(get_distribution_3dangle(mouse))
    data.append(features)
    return data

# f2
def getFs_clf(data):
    mouse=data[1]
    features=[]
    features.extend(get_t_T(mouse))
    data.append(features)
    # data[1]=features
    return data

trains = rdd_train.map(splitx).filter(filterXNP).map(getFs_shape).map(getFs_clf)
trains =trains.collect()
vector_shape=[]
lables_shape=[]

for d in trains:
    idx=d[0]
    f1=d[4]
    if idx in range(2601):
        continue
    if idx in range(2651,2701) or idx in range(2801,2901):
        label=1
    else:
        label=2
    vector_shape.append(f1)
    lables_shape.append(label)

vector_shape=np.array(vector_shape)    
lables_shape=np.array(lables_shape) 
scaler_shape = preprocessing.StandardScaler().fit(vector_shape)
vector_shape = preprocessing.scale(vector_shape)
clf_shape=SVC(C=0.5)
clf_shape.fit(vector_shape,lables_shape)

# classifier #################################

vector_shape1=[]
lables_shape1=[]

vector_shape2=[]
lables_shape2=[]

for d in trains:
    idx=d[0]
    f2=d[5]
    label=1
    if idx in range(2601):
        vector_shape1.append(f2)
        vector_shape2.append(f2)
        lables_shape1.append(label)
        lables_shape2.append(label)
        continue
    label=0
    if idx in range(2651,2701) or idx in range(2801,2901):
        vector_shape1.append(f2)
        lables_shape1.append(label)
    else:
        vector_shape2.append(f2)
        lables_shape2.append(label)

vector_shape1=np.array(vector_shape1)    
lables_shape1=np.array(lables_shape1) 
scaler_shape1 = preprocessing.StandardScaler().fit(vector_shape1)
vector_shape1 = preprocessing.scale(vector_shape1)
clf_shape1=SVC(C=0.5)
clf_shape1.fit(vector_shape1,lables_shape1)

vector_shape2=np.array(vector_shape2)    
lables_shape2=np.array(lables_shape2) 
scaler_shape2 = preprocessing.StandardScaler().fit(vector_shape2)
vector_shape2 = preprocessing.scale(vector_shape2)
clf_shape2=SVC(C=0.5)
clf_shape2.fit(vector_shape2,lables_shape2)        


def testDs(data,conf):
    scalar=conf["scalar"]
    clf=conf["clf"]    
    scalar1=conf["scalar1"]
    clf1=conf["clf1"]    
    scalar2=conf["scalar2"]
    clf2=conf["clf2"]

    mouse=data[1]
    if get_sharp_angle(mouse)==True:
        return True

    f1=data[4]
    f1=scalar.transform([f1])
    r=clf.predict(f1)

    f2=data[5]
    if r[0]==1: # class1
        s1=scalar1.transform([f2])
        s1r=clf1.predict(s1)
        if s1r[0]==0:
            return True
        else:
            return False
    else:
        s2=scalar2.transform([f2])
        s2r=clf2.predict(s2)
        if s2r[0]==0:
            return True
        else:
            return False

conf={"scalar":scaler_shape,"clf":clf_shape, \
"scalar1":scaler_shape1,"clf1":clf_shape1, \
"scalar2":scaler_shape2,"clf2":clf_shape2 }

output = rdd_all.map(splitx).filter(filterXNP).map(getFs_shape).map(getFs_clf) \
        .filter(lambda d:testDs(d,conf)).map(lambda d:d[0])

# output =output.collect()
num=output.count()
print "l \\ shape numbers:",num

output.saveAsTextFile(output_path)