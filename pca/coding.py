# coding=utf-8
import numpy as np 
import datadraw as dw

def get_k(lbdnn,p):
    # get the k of top lbdnn
    pass 
    k=0
    return k

def pca(X,p=0.99):
    n,m=X.shape
    # TODO1: get the means
    # TODO2: get the cov =XXT/m
    # TODO3: get the lambda with np.linalg.eig(np.mat())
    # TODO4: get k top of lambda 
    # TODO5: get top k vector as k_P
    # TODO6: get Y＝PT*X
    # TODO7: get newX = P*PT*X +means
    pass
    Y=[]
    newX=[] 
    return Y,newX 

if __name__ == '__main__':
    X=np.array([[-1,-1,0,2,0],[-2,0,0,1,1]],dtype=np.float)
    Y,newX=pca(X,0.8)
    print Y
    print newX
    # dw.draw(X,Y,newX)

   