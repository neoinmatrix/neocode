import sys
from pyspark import SparkContext
output = sys.argv[1]
input = sys.argv[2]
sc = SparkContext(appName="test")
rdd = sc.textFile(input)

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
    return [items[0],[mtr_x,mtr_y,mtr_t]]

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

result = rdd.map(splitx).map(filterx).filter(lambda f:f[1]).map(lambda f:f[0])


# xpn=rdd.map(splitx).filter(lambda f:get_X_PN_isMachine(f[1])).map(lambda f:f[0])
# print "xpn:",len(xpn.collect())
# x437=rdd.map(splitx).filter(lambda f:get_437(f[1])).map(lambda f:f[0])
# print "x437:",len(x437.collect())
# sharp=rdd.map(splitx).filter(lambda f:get_sharp_angle(f[1])).map(lambda f:f[0])
# print "sharp:",len(sharp.collect())

result.saveAsTextFile(output)