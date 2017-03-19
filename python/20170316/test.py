# coding=utf-8
# print "hello"
import numpy as np
import sys
# print np
# a=np.array([1,2,3,4,5,6])
# print a 
# print sys.getsizeof(a)

# print a.shape[0]
# print a[0].dtype

# a.shape=2,-1
# a=np.arange(0.1,10,2)
# a=np.linspace(0,10,50,False,True,np)
# a=np.random.rand(3,2,1)
# a=np.random.rand(3,3)

a=np.arange(1,10).reshape(3,3)
b=np.arange(1,10).reshape(3,3)
# c=np.dot(a,b.T)
# print a
# print b
# print b.T
# print c
c=np.outer(a,b)
d=np.inner(a,b)
print c
print d
np.save("c.txt",c)
np.save("d.txt",d)
