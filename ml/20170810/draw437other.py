import sys
from pyspark import SparkContext
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

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
    if len(r_arr)>1 and mean>30.0:
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
    angles=getfivex(t,range(len(t)-5,len(t)),5)
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
    # features.extend([mouse[0][0],mouse[1][0],mouse[2][0]])
    features.extend(get_distribution_3dangle(mouse))
    data.append(features)
    return data

# f2
def getFs_clf(data):
    mouse=data[1]
    features=[]
    features.extend(getf(mouse))
    features.extend(getft(mouse))
    features.extend(get_t_T(mouse))
    data.append(features)
    # data[1]=features
    return data

def getFs(data):
    mouse=data[1]
    features=[]
    features.extend(getf(mouse))
    features.extend(getft(mouse))
    features.extend(get_distribution_3dangle(mouse))
    features.extend(get_t_T(mouse))
    data.append(features)
    return data

# output = rdd_all.map(splitx).filter(filterXNP).map(getFs) \
#         .filter(lambda d:testDs(d,conf)).map(lambda d:d[0])

def analyst(data):
    mouse=data[1]
    if mouse[0][0]<437:
        if mouse[0][0]<450 and  mouse[0][0]>250:
            if mouse[1][0]<2650 and  mouse[1][0]>2450:
                if get_sharp_angle(mouse)==True:
                    return [1,mouse]
                else:
                    return [12,mouse]
        return [2,mouse]
    else:
        if mouse[0][0]<600:
            if mouse[1][0]<2500:
                # return [3,mouse]
                if get_sharp_angle(mouse)==True:
                    return [3,mouse]
                else:
                    return [32,mouse]
        return [4,mouse]

def getitems(data):
    if data[0]==4:
        return True
    else:
        return False

def getdrawdata(data):
    mouse=data[1]
    return "0 "+str(mouse[0][0])+" "+str(mouse[1][0])

output = rdd_all.map(splitx).filter(filterXNP).map(analyst).filter(getitems).map(getdrawdata)

# data=output.collect()
# print data
# num=output.countByValue().items()
# print "l \\ shape numbers:",num

output.saveAsTextFile(output_path)