import sys
from pyspark import SparkContext
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing

output_path = sys.argv[1]
input_train = sys.argv[2]
input_all = sys.argv[3]

sc = SparkContext(appName="train")
rdd_train = sc.textFile(input_train)
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

def get_xpn_mac(mouse,inverse=False): # tempor
    x=mouse[0]
    n=len(x)
    for i in range(1,n):
        if x[i-1]>x[i]:
            if inverse==True:
                return True
            return False
    if inverse==True:
        return False
    return True

def get_nd(d,idxx):
    n=len(d)
    for i in range(len(idxx)):
        if idxx[i]<0:
            idxx[i]=0
        if idxx[i]>(n-1):
            idxx[i]=n-1
    d=np.array(d)
    return d[idxx].flatten()

# features =========================
def get_angle_sharp(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])

    angle_dist=np.zeros(3,dtype=np.float)
    sharp=np.array([0])
    for i in range(1,n):
        if i+1>=n:
            break
        else:
            vx1=x[i+1]-x[i]
            vy1=y[i+1]-y[i]
            # vt1=t[i+1]-t[i]
            vx2=x[i]-x[i-1]
            vy2=y[i]-y[i-1]
            # vt2=t[i]-t[i-1]
            dt=t[i+1]-t[i-1]
            angle=vx1*vx2+vy1*vy2

            # angle3d=angle+vt1*vt2
            if dt==0:
                continue
            r1=(vx1**2+vy1**2)**0.5
            r2=(vx2**2+vy2**2)**0.5

            if r1==0 or r2==0:
                continue
            angle/=r1
            angle/=r2
            # print angle,dt,r1,r2
            if angle>-1.1 and angle<=-0.3:
                angle_dist[0]+=1
            elif angle>-0.3 and angle<=0.3:
                angle_dist[1]+=1
            elif angle>0.3:
                angle_dist[2]+=1
            if angle<0:
                rr=r1 if r1<r2 else r2
                sharp=np.append(sharp,rr)
    
    feat=np.array([])
    feat=np.append(feat,angle_dist)
    feat=np.append(feat,float(len(sharp)))
    feat=np.append(feat,sharp.mean())
    feat=np.append(feat,sharp.max())
    return feat.flatten()

def get_diff(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])
    def statistic(tmp):
        return [len(tmp),tmp.min(),tmp.max(),tmp.mean(),tmp.std()]
    dx=np.array([0])
    dy=np.array([0])
    dt=np.array([0])
    for i in range(1,n):
        dx=np.append(dx,x[i]-x[i-1])
        dy=np.append(dy,y[i]-y[i-1])
        dt=np.append(dt,t[i]-t[i-1])
    feat=np.array([])
    feat=np.append(feat,statistic(dx))
    feat=np.append(feat,statistic(dy))
    feat=np.append(feat,statistic(dt))
    return feat.flatten()

def get_cxcy(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])
    # x=x/x.max()
    # y=y/y.max()
    # t=t/t.max()
    tmpx=x[0]
    tmpy=y[0]
    tmpt=0
    countx=0.0
    county=0.0
    tarr=np.array([0])
    for i in range(1,n):
        if tmpx!=x[i]:
            tmpx=x[i]
            countx+=1
        if tmpy!=y[i]:
            tmpy=y[i]
            county+=1
            tarr=np.append(tarr,tmpt)
            tmpt=0
        tmpt+=(t[i]-t[i-1])
    if n>1:
        startdt=(t[1]-t[0])/t[-1]
    else:
        startdt=0

    feat=np.array([])
    feat=np.append(feat,[startdt,countx/float(n),county/float(n),n])
    feat=np.append(feat,[tarr.max(),tarr.mean(),tarr.std()])
    return feat.flatten()

def get_action_shape(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])

    tmpy=y[0]
    county=0.0
    vxt=np.array([0])
    for i in range(1,n):
        vx=x[i]-x[i-1]
        dt=t[i]-t[i-1]
        if dt==0:
            continue
        angle=vx/dt
        vxt=np.append(vxt,angle)
        if tmpy!=y[i]:
            county+=1
            tmpy=y[i]
    idx=range(0,5)
    start=get_nd(vxt,idx)
    idx=range(len(vxt)-5,len(vxt))
    over=get_nd(vxt,idx)

    feat=[]
    feat=np.array([])
    if n>1:
        startdt=(t[1]-t[0])/t[-1]
    else:
        startdt=0
    feat=np.append(feat,startdt)
    feat=np.append(feat,county/float(n))
    feat=np.append(feat,[start.mean(),start.max()])
    feat=np.append(feat,[over.mean(),over.max()])

    return feat.flatten()

def get_towards(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])

    tmpx=x[0]
    tmpy=y[0]
    xy=[]
    for i in range(1,n):
        if tmpx==x[i] or tmpy==y[i]:
            continue
        xy.append([x[i],y[i]])
        tmpx=x[i]
        tmpy=y[i]
    tan_arr=np.array([0])
    for i in range(1,len(xy)):
        sx,sy=xy[i-1]
        ex,ey=xy[i]
        tan=(ey-sy)/(ex-sx)
        tan_arr=np.append(tan_arr,tan)

    tmptan=tan_arr[0]
    change=np.array([0,0,0])
    for i in range(1,len(tan_arr)):
        if tan_arr[i]>0:
            change[0]+=1
        else:
            change[1]+=1
        if tan_arr[i]*tmptan<0:
            change[2]+=1
        tmptan=tan_arr[i]
    feat=[]
    feat=np.array([])
    if n>1:
        startdt=(t[1]-t[0])/t[-1]
    else:
        startdt=0
    # feat=np.append(feat,startdt)
    feat=np.append(feat,change/float(n))
    feat=np.append(feat,float(n))
    feat=np.append(feat,len(xy))

    return feat.flatten()

def get_all(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(mouse[0])
    # x=x/max(x)
    # y=y/max(y)
    # t=t/max(t)
    
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
    lastfive=get_nd(vxt,idx)
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
    t3=get_nd(t,idx)
    feat.extend(t3)

    # if  len(mouse[2])>1:
    #     feat.append(1.0 if mouse[2][1]>500 else 0.0)
    # else:
    #     feat.append(0.0)

    # first five angle is concave convex
    xfive=get_nd(x,range(0,5))
    tfive=get_nd(t,range(0,5))
    ax=xfive[4]-xfive[0]
    at=tfive[4]-tfive[0]
    mx=(xfive[4]+xfive[0])/2
    mt=(tfive[4]+tfive[0])/2
    bx=mx-xfive[2]
    bt=mt-tfive[2]
    angle=ax*bx+at*bt
    feat.append(angle)

    return np.array(feat).flatten()

def get_feats(data):
    mouse=data[1]
    data.append(get_angle_sharp(mouse))
    data.append(get_diff(mouse))
    ftmp=np.array([])
    ftmp=np.append(ftmp,get_towards(mouse))
    ftmp=np.append(ftmp,get_cxcy(mouse))
    ftmp=np.append(ftmp,get_action_shape(mouse))
    # ftmp=np.append(ftmp,get_all(mouse))
    data.append(ftmp.flatten())
    data.append(get_all(mouse))
    return data

# data
trains = rdd_train.map(splitx) \
    .filter(lambda d:get_xpn_mac(d[1])).map(get_feats)
trains =trains.collect()

# get features
vtrs={"sharp":[],"diff":[],"middle":[],"all":[]}
lbs={"sharp":[],"diff":[],"middle":[],"all":[]}
for d in trains:
    idx=d[0]
    label=d[3]
    f={"sharp":4,"diff":5,"middle":6,"all":7}
    for k,v in f.items():
        f[k]=d[v]

    if idx in range(2601):
        for k in vtrs:
            vtrs[k].append(f[k])
            lbs[k].append(label)
    else:
        k=""
        if idx in range(2901,3001):
            k="sharp"
            vtrs[k].append(f[k])
            lbs[k].append(label)
        if idx in range(2651,2901):
            k="diff"
            vtrs[k].append(f[k])
            lbs[k].append(label)
        if idx in range(2601,2801):
            k="middle"
            vtrs[k].append(f[k])
            lbs[k].append(label)
        if idx in range(2601,3001):
            k="all"
            vtrs[k].append(f[k])
            lbs[k].append(label)
    

np.set_printoptions(formatter={'float':lambda x: "%5.2f"%float(x)})   

# training 
scalar={"sharp":[],"diff":[],"middle":[],"all":[]}
clf={"sharp":[],"diff":[],"middle":[],"all":[]}
for k in scalar:
    vector=np.array(vtrs[k])
    label=np.array(lbs[k])
    scalar[k]= preprocessing.StandardScaler().fit(vector)
    vector = preprocessing.scale(vector)    
    clf[k]= SVC(C=2.5)
    clf[k].fit(vector,label)

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

def testDs(d,conf):
    mouse=d[1]
    scalar=conf["scalar"]
    clf=conf["clf"]
    f={"sharp":4,"diff":5,"middle":6,"all":7}
    for k,v in f.items():
        f[k]=d[v]
    dbg=conf['debug']
    # if inarea(mouse):
    #     if dbg==False:
    #         return True
    #     return "inarea"
    # if inarea2(mouse):
    #     if dbg==False:
    #         return True
    #     return "inarea2"
    if inarea(mouse)==False:
        return False
    key="sharp"
    feat=f[key]
    feat=scalar[key].transform([feat])
    rr=clf[key].predict(feat)
    if rr[0]==0:
        return True
    return False
    # keys=["sharp","diff","middle","all"]
    # for key in keys:
    #     feat=f[key]
    #     feat=scalar[key].transform([feat])
    #     rr=clf[key].predict(feat)
    #     if rr[0]==0:
    #         if inarea(mouse):
    #             if dbg==False:
    #                 return True
    #             return "inarea"+key
    #         if dbg==False:
    #             return True
    #         return key
   
    # if dbg==False:
    #     return False
    # if inarea2(mouse):
    #     return "machine_inarea2"
    # return "machine"

conf={"scalar":scalar,"clf":clf,'debug':True}
output = rdd_all.map(splitx).filter(lambda d:get_xpn_mac(d[1],True)) \
    .map(get_feats).filter(lambda d:testDs(d,conf)).map(lambda d:d[1])
    # .map(get_feats).map(lambda d:testDs(d,conf))
    # .map(get_feats).map(lambda d:testDs(d,conf))


num=output.count()
print num
print output.collect()
# num=output.countByValue().items()
# print " classifier results:",num

# print output.collect()
# output.saveAsTextFile(output_path)