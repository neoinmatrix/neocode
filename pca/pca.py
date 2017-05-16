# coding=utf-8
import numpy as np 
import matplotlib.pyplot as plt

def get_k(lbdnn,p):
    sort=np.sort(lbdnn) 
    sort=sort[-1::-1] 
    sumall=sum(sort)*p
    tmpsum=0
    k=0
    for v in sort:
        tmpsum+=v
        k+=1
        if tmpsum>=sumall:
            break 
    return k

def pca(X,p=0.99):  
    n,m=X.shape
    means=np.mean(X,axis=1) 
    for i in range(n):
        X[i]=X[i]-means[i]
    cov=np.dot(X,X.T)/m
    lbdnn,P=np.linalg.eig(np.mat(cov))

    k=get_k(lbdnn,p)
    sort_lbd=np.argsort(lbdnn)   
    sort_lbd=sort_lbd[-1::-1]

    sort_lbd=sort_lbd[:k]
    k_P=P[:,sort_lbd]
    Y=k_P.T*X
    newX= k_P*k_P.T*X
    for i in range(len(means)):
        newX[i]+=means[i]
    return Y,newX 

def draw(X,Y,newX):
    plt.scatter(X[0,:],X[1,:])
    plt.scatter(Y/(2**0.5),[0]*Y.shape[1],c='r',marker='<')
    plt.scatter(newX[0,:],newX[1,:],c='y',marker='x')

    plt.grid(True)
    plt.plot([-2.5,0.5,1,2.5], [-2.5,0.5,1,2.5], 'g-')
    plt.axis([-2.5,2.5,-2.5 ,2.5])
    plt.yticks([i for i in range(-3,4,1)])
    plt.xticks([i for i in range(-3,4,1)])
    plt.show()

if __name__ == '__main__':

    X=np.array([[-1,-1,0,2,0],[-2,0,0,1,1]],dtype=np.float)
    Y,newX=pca(X,0.8)
    print Y
    print newX

    # draw(X,Y,newX)