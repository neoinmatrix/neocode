# coding=utf-8
import numpy as np 

# get 3d vector  the angle number between two vector
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

# find sharp one 
def get_sharp_angle(mouse):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    xn=len(mouse[0])
    angle_arr=[]
    r_arr=[]
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

# if x toward change must be manual tracking 
# x change then t change so ingore t change
def get_X_PN(mouse):
    x=mouse[0]
    n=len(x)
    for i in range(1,n):
        if x[i-1]>x[i]:
            return True
    return False

#========common=================
# create save string 
def createstr(data):
    tmp=''
    for v in data:
        tmp+=str(v)+"\n"
    return tmp

# get value in range of sequences
def getfivex(d,idxx,num=5):
    xn=len(d)
    for i in range(num):
        if idxx[i]<0:
            idxx[i]=0
        if idxx[i]>(xn-1):
            idxx[i]=xn-1
    d=np.array(d)
    return d[idxx].tolist()

# summary_print_save
def getSummary(mclass,rclass,path='./data/'):
      # ===print=========================================  
    tmp1=''
    print "summary:"
    for i in range(len(mclass)):
        if len(mclass[i])>0:
            tmp1+="%d "%len(mclass[i])
    print tmp1
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
        with open(path+"m_%d.txt"%i,'w') as f:
            f.write(savestr)
        
    for i in range(len(rclass)):
        if len(rclass[i])==0:
            continue
        savestr=createstr(rclass[i])
        with open(path+"r_%d.txt"%i,'w') as f:
            f.write(savestr)
