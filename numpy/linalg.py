import numpy as np 

a=np.zeros([3,3])
idx=np.array([0,1,2])
a[idx,idx]=idx+1


# print a
# print np.linalg.inv(a)
# print a*np.linalg.inv(a)

a=np.array([[1,2],[1,2]])
b=np.array([[3,4],[5,6]])

# print a
# print b
# print a*b
# print np.dot(a,b)

c=np.array([[1,3],[2,4]])
d=np.array([[1],[1]])
# print np.linalg.solve(c,d)
a=np.zeros([3,3])
idx=np.array([0,1,2])
a[idx,idx]=idx+1
# print c
# print np.linalg.eig(a)
# print np.trace(a)
# print np.linalg.matrix_rank(a)