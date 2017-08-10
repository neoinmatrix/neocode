import sys
from pyspark import SparkContext
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
# from sklearn.ensemble import RandomForestClassifier

output_path = sys.argv[1]
inputx = sys.argv[2]
input_all = sys.argv[3]

sc = SparkContext(appName="train")
rdd_train = sc.textFile(inputx)
rdd_all = sc.textFile(input_all)

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
    feat.extend(lastfive)
    lastfive=np.array(lastfive)
    feat.extend([lastfive.std(),lastfive.mean()])

    # idx=range(0,5)
    # beginfive=getfivex(vxt,idx)
    # beginfive=np.array(beginfive)
    # feat.extend([beginfive.std(),beginfive.mean()])

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

def get_all_statistic(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    def getanalyst(tmp):
        return [tmp.min(),tmp.max(),tmp.mean(),tmp.std()]
    analyst=[]
    analyst.extend(getanalyst(x))
    analyst.extend(getanalyst(y))
    analyst.extend(getanalyst(t))
    return analyst

def get_t_entropy(mouse):
    t=mouse[2]
    n=len(t)
    ta=t[-1]
    if ta==0:
        ta=1.0
    entropy=0.0
    for i in range(1,n):
        dt=t[i]-t[i-1]
        if dt==0:
            dt=1.0
        if dt<0:
            entropy=0.0
            continue
        p=float(dt)/float(ta)
        p=p if p>0 else 1.0
        entropy+=np.log(p)*p
    entropy=entropy if entropy<0 else 0.0
    return [entropy]

def get_speed(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    xn=len(mouse[0])
    vx=[0.0]
    vy=[0.0]
    for i in range(1,xn):
        vspeed=(float(x[i])-float(x[i-1]))
        vyspeed=(float(y[i])-float(y[i-1]))
        dt=float(t[i])-float(t[i-1])
        if dt==0:
            continue
        vspeed/=dt
        vyspeed/=dt
        vx.append(vspeed)
        vy.append(vyspeed)
    vx=np.array(vx)
    vy=np.array(vy)
    return [vx.max(),vx.min(),vx.mean(),vy.max(),vy.min(),vy.mean()]
 

def get_feats(data):
    mouse=data[1]
    features=[]
    features.extend([mouse[0][0],mouse[2][0]])
    features.extend(get_shape(mouse))
    features.extend(get_distx(mouse))
    # features.extend(get_all_statistic(mouse))
    features.extend(get_t_entropy(mouse))
    features.extend(get_speed(mouse))
    data.append(features)
    return data

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
    if len(r_arr)>1 and mean>30.0:
        return True
    else:
        return False


trains = rdd_train.map(splitx).filter(filter_xnp).map(get_feats)
trains =trains.collect()
vector_shape=[]
lables_shape=[]

vector_hard=[]
lables_hard=[]

for d in trains:
    idx=d[0]
    label=d[3]
    f1=d[4]
    # f2=d[5]

    vector_shape.append(f1)
    lables_shape.append(label)

    # vector_hard.append(f2)
    # lables_hard.append(label)

vector_shape=np.array(vector_shape)    
lables_shape=np.array(lables_shape) 
scalar_easy = preprocessing.StandardScaler().fit(vector_shape)
vector_shape = preprocessing.scale(vector_shape)

# vector_hard=np.array(vector_hard)    
# lables_hard=np.array(lables_hard) 
# scalar_hard = preprocessing.StandardScaler().fit(vector_hard)
# vector_hard = preprocessing.scale(vector_hard)

clf_easy=SVC(C=1.5)
clf_easy.fit(vector_shape,lables_shape)

# clf_hard=SVC(C=1.5)
# clf_hard.fit(vector_hard,lables_hard)

def inarea(mouse):
    if mouse[0][0]>=437:
        if mouse[0][0]<600:
            if mouse[1][0]<2500:
                return True
    return False

def inarea2(mouse):
    if mouse[0][0]<370:
        if mouse[0][0]>250:
            if mouse[1][0]<2650 and mouse[1][0]>2450:
                return True
    return False

def testDs1(data,conf):
    mouse=data[1]
    if get_sharp_angle(mouse)==True:
        return 1
    if inarea(mouse):
        return 2   
    if inarea2(mouse):
        scalar=conf["scalar_easy"]
        clf=conf["clf_easy"]    
        f1=data[4]
        f1=scalar.transform([f1])
        r=clf.predict(f1)
        if r[0]==0: # machine
            return 3 
        else:
            return 4

    scalar=conf["scalar_hard"]
    clf=conf["clf_hard"]    
    f2=data[5]
    f2=scalar.transform([f2])
    r=clf.predict(f2)
    if r[0]==0: # machine
        return 5
    else:
        return 6
  
def testDs(data,conf):
    mouse=data[1]
    if get_sharp_angle(mouse)==True:
        return True
    if inarea(mouse):
        return True  
    scalar=conf["scalar_easy"]
    clf=conf["clf_easy"]    
    f1=data[4]
    f1=scalar.transform([f1])
    r=clf.predict(f1)
    if r[0]==0: # machine
        return True 
    else:
        return False

    # if inarea2(mouse):
    #     # return True
    #     scalar=conf["scalar_easy"]
    #     clf=conf["clf_easy"]    
    #     f1=data[4]
    #     f1=scalar.transform([f1])
    #     r=clf.predict(f1)
    #     if r[0]==0: # machine
    #         return True 
    #     else:
    #         return False

    # scalar=conf["scalar_hard"]
    # clf=conf["clf_hard"]    
    # f2=data[5]
    # f2=scalar.transform([f2])
    # r=clf.predict(f2)
    # if r[0]==0: # machine
    #     return True
    #     # return 5
    # return False
    # # else:
    #     return 6

conf={"scalar_easy":scalar_easy,"clf_easy":clf_easy }
# ,
# "scalar_hard":scalar_hard,"clf_hard":clf_hard

output = rdd_all.map(splitx).filter(filter_xnp).map(get_feats).filter(lambda d:testDs(d,conf)).map(lambda d:d[0])
 #.map(lambda d:testDs(d,conf))
        
        

num=output.count()
# num=output.countByValue().items()
print "l \\ shape numbers:",num

# print output.collect()

output.saveAsTextFile(output_path)