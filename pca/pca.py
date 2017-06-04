# coding=utf-8
import numpy as np 
import datadraw as dw

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

def pca(X,p=0.99,getnew=False):  
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
    newX=''
    if getnew==True:
        newX= k_P*k_P.T*X
        for i in range(len(means)):
            newX[i]+=means[i]
    return Y,newX 

if __name__ == '__main__':
    X=np.array([[-1,-1,0,2,0],[-2,0,0,1,1]],dtype=np.float)
    Y,newX=pca(X,0.8,True)
    print Y
    print newX
    dw.draw(X,Y,newX)