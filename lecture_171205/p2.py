# coding:utf-8
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

isdebug = True
isdebug = False
X=[]
Mu=[]
Expectations=[]


def ini_data(alpha,mu,N=50):
    x=[]
    for i in xrange(0,N):
        if np.random.random(1) <alpha:
            tmp = np.random.normal()*mu[1] + mu[0]
        else:
            tmp = np.random.normal()*mu[3] + mu[2]
        x.append(tmp)
    return np.array(x)

def phi(x,u,q):
    up=math.exp((-1/(2*(float(q**2))))*(float(x-u)**2))
    down=1/(math.sqrt(2*math.pi)*q)
    return up*down

def e_step(Mu,alpha,k,data):
    # print 0.126501749999/(0.506006999997+0.126501749999)
    r=np.zeros([len(data),k],dtype=np.float)
    for j in range(len(data)):
        m1=alpha*phi(data[j],Mu[0],Mu[1])
        m2=(1-alpha)*phi(data[j],Mu[2],Mu[3])
        # print m1,m2
        # exit()
        # print m1,m2
        # print m1/(m1+m2)
        r[j,0]=m1/(m1+m2)
        r[j,1]=m2/(m1+m2)
    # print r
    # exit()
    return r

def m_step(r,Mu_f,data):
    Mu=np.array([0,0,0,0],dtype=np.float)
    Mu[0]=sum(r[:,0]*data)/sum(r[:,0])
    Mu[1]=sum(r[:,0]*(data-Mu_f[0])**2)/sum(r[:,0])
    Mu[1]=math.sqrt(Mu[1])

    Mu[2]=sum(r[:,1]*data)/sum(r[:,1])
    Mu[3]=sum(r[:,1]*(data-Mu_f[2])**2)/sum(r[:,1])
    Mu[3]=math.sqrt(Mu[3])

    alpha=sum(r[:,0])/len(r[:,0])
    return alpha,Mu

def run(data,k,iter_num,Epsilon):
    Mu=np.array([-37.,70.,70.,20.])
    alpha=0.5
    for i in range(iter_num):
        Old_Mu = copy.deepcopy(Mu)
        r=e_step(Mu,alpha,k,data)
        alpha,Mu=m_step(r,Mu,data)
        print i,alpha,Mu
        if sum(abs(Mu-Old_Mu)) < Epsilon:
            break

if __name__ == '__main__':
    x=np.array([-67,-48,6,8,14,16,23,24,28,29,41,49,56,60,75],dtype=np.float)
    # x=ini_data(0.3,[0,10,50,15])
    # x=sorted(x)
    # print x
    np.set_printoptions(precision=4)
    run(x,2,100,0.0001)
    # plt.hist(X[0,:],35)
    # plt.hist(x,30)
    # plt.show()