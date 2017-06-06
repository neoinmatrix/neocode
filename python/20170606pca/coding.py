# coding=utf-8
import numpy as np 
import datadraw as dw

def get_k(lbdnn,p):
    # get the k of top lbdnn
    sort=np.sort(lbdnn)
    sort=sort[-1::-1]
    sumall=sum(sort)*p
    sumtmp=0.0
    k=0
    for v in sort:
        k+=1
        sumtmp+=v
        if sumtmp>=sumall:
            break
    return k

def pca(X,p=0.99):
    n,m=X.shape
    # TODO1: get the means
    means=np.mean(X,axis=1)
    for i in range(n):
        X[i]-=means[i]
    # print X
    # TODO2: get the cov =XXT/m
    cov=np.dot(X,X.T)/m
    # print cov
    # TODO3: get the lambda with np.linalg.eig(np.mat())
    lbd,P=np.linalg.eig(np.mat(cov))
    # print lbd,P
    # TODO4: get k top of lambda 
    k=get_k(lbd,p)
    # print k
    # TODO5: get top k vector as k_P
    idx=np.argsort(lbd)
    idx=idx[-1::-1]
    idx=idx[:k]
    # print idx
    n_P=P[:,idx]
    # print n_P
    # TODO6: get Y＝PT*X
    Y=n_P.T*X
    newX=n_P*n_P.T*X
    # TODO7: get newX = P*PT*X +means
    for i in range(n):
        newX[i]+=means[i]
    return Y,newX 

if __name__ == '__main__':
    X=np.array([[-1,-1,0,2,0],[-2,0,0,1,1]],dtype=np.float)
    Y,newX=pca(X,0.8)
    # print Y
    # print newX
    # dw.draw(X,Y,newX)

   