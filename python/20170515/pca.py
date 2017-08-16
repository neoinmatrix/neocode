import numpy as np 

def zeroMean(X):        
    mean=np.mean(X,axis=0)     #按列求均值，即求各个特征的均值  
    X_mean=X-mean  
    return X_mean,mean

def pca(X,percentage=0.99):  
    X,mean=zeroMean(X)  
    cov=np.cov(X,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本  
    lambdaVal,P=np.linalg.eig(np.mat(cov))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量  
    n=percentage2n(lambdaVal,percentage)          #要达到percent的方差百分比，需要前n个特征向量  
    sortlambadaVal=np.argsort(lambdaVal)            #对特征值从小到大排序  
    n_lambadaIndex=sortlambadaVal[-1:-(n+1):-1]   #最大的n个特征值的下标  
    n_P=P[:,n_lambadaIndex]                       #最大的n个
    Y=X*n_P                             #低维特征空间的数据  
    newX=(Y*n_P.T)+mean    #重构数据  
    return Y,newX 
data=np.array([[-1,-1,0,2,0],[-2,0,0,1,1]],dtype=np.float)
print data