import numpy as np 

def zeroMean(X):        
    means=np.mean(X,axis=1)
    X_mean=X
    for i in range(len(X_mean)):
        X_mean[i]=X_mean[i]-means[i]
    return X_mean,means

def percentage_n(lambdaVal,percentage):
    sortArray=np.sort(lambdaVal) 
    sortArray=sortArray[-1::-1] 
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

def pca(X_raw,percentage=0.99):  
    X,means=zeroMean(X_raw) 
    cov=np.dot(X,X.T)/len(X[0])
    lambdaVal,P=np.linalg.eig(np.mat(cov))
    k=percentage_n(lambdaVal,percentage)
    sortlambadaVal=np.argsort(lambdaVal)   
    k_lambadaIndex=sortlambadaVal[-1:-(k+1):-1]  
    k_P=P[:,k_lambadaIndex] 

    Y=k_P.T*X_raw  
    print  k_P.T*k_P
    newX=k_P*k_P.T*X_raw 
    for i in range(len(newX)):
        newX[i]+=means[i]
    print newX
    exit()
    return Y,newX 
data=np.array([[-1,-1,0,2,0],[-2,0,0,1,1]],dtype=np.float)
# print data
Y,newX=pca(data,0.8)