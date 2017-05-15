import numpy as np 

def percentage_n(lbd,percentage):
    sort=np.sort(lbd) 
    sort=sort[-1::-1] 
    sumall=sum(sort)*percentage
    for i in range(1,len(lbd)+1):
        if sum(sort[:i])>=sumall:
            break
    return i 

def pca(A,percentage=0.99):
    dims,nums=A.shape
    means=np.mean(A,axis=1)
    Amean=A
    for i in range(dims):
        Amean[i]=Amean[i]-means[i]
    covA=np.dot(A,A.T)/float(nums)
    lbd,U=np.linalg.eig(np.mat(covA))
    k=percentage_n(lbd,percentage)
    index_lbd=np.argsort(lbd)
    index_lbd=index_lbd[-1::-1]
    k_index=index_lbd[:k]
    k_U=U[:,k_index] 

    Y=k_U.T*A  
    # print Y.shape
    # print  k_P.T*k_P
    newX=k_U.T*A*k_U 
    for i in range(len(newX)):
        newX[i]+=means[i]
    print newX
    exit()
    return Y,newX 
A=np.array([[-1,-1,0,2,0],[-2,0,0,1,1]],dtype=np.float)

# print data
Y,newX=pca(A,0.8)
# print len([])
# a=A[0]
# print a
# b=np.argsort(a)
# print b
# print b[-1::-1]


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
# ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
# plt.show()