import numpy as np
from numpy import newaxis

a=np.logspace(1,3,num=3)
np.set_printoptions(formatter={"float":lambda x:"%.10f"%float(x)})
print a
# import matplotlib.pyplot as plt
# plt.plot(range(len(a)),a)
# plt.show()
# a=np.zeros([3])
# b=np.ones([3])
# a=np.c_[a]
# print id(a)
# print id(b)
# c=a.view()
# d=a.copy()
# print id(c)
# print id(d)
# e=a
# print id(e)
# print a
# print np.vsplit(a,3)[0][0]
# print np.column_stack([a[:,newaxis],b[:,newaxis]])
# print np.column_stack([a,b])
# print np.vstack([a,b])
# print np.hstack([a,b])
# print np.r_[a,1,2]
# print a[:,newaxis]
# print np.c_[a,b]

# print a[:,newaxis]
# print np.hstack([a,3]) 
# a=np.arange(10)
# b=np.arange(10)
# c=np.vstack([a,b])
# d=np.hstack([a,b])
# print c
# print d


# print a.reshape([2,-1])
# a=a.reshape([2,4])
# print a.resize((2,4))
# def f(x,y):
#     return x*10+y
# b=np.fromfunction(f,(5,2))
# print a,b

# b=range(10)*3
# print b
# a=np.arange(10)*3
# print a
# a=np.arange(15).reshape([3,5])
# print a.cumsum(axis=0)
# print np.linspace(0,np.pi,3)

# a=np.arange(15).reshape([3,5])
# print a
# print a.shape
# print a.ndim
# print a.dtype
# print a.dtype.name
# print a.itemsize
# print a.size
# print type(a)
# print np.empty([2,3])

# print np.arange(1,10,2)
# print np.arange(1,10,0.1)
# print np.linspace(0,np.pi,9)
# print np.arange(1,10)

# np.set_printoptions(threshold=np.nan)
# print np.arange(10000).reshape([100,100])

