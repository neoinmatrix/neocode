import numpy as np
import math

def e(data,pi,p,q):
    u=[]
    for v in data:
        up=pi*math.pow(p,v)*math.pow((1-p),(1-v))
        down=pi*math.pow(p,v)*math.pow((1-p),(1-v))+(1-pi)*math.pow(q,v)*math.pow((1-q),(1-v))
        u.append(float(up/down))
    return np.array(u)

def m(data,u):
    pi=sum(u)/len(u)
    p=sum(u*data)/sum(u)
    q=sum((1-u)*data)/sum(1-u)
    return pi,p,q

data=np.array([1,1,0,1,0,0,1,0,1,1])

# data
pi=0.5
p=0.5
q=0.5
# u=e(data,pi,p,q)
# pi,p,q=m(data,u)
# print pi,p,q

pi=0.46
p=0.55
q=0.67

for i in range(10):
    u=e(data,pi,p,q)
    pi,p,q=m(data,u)
    print  pi,p,q