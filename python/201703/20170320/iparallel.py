# coding=utf-8
# using ipyparallel to calc the data

import ipyparallel as ipp
import time
c = ipp.Client()

def runx():
    for i in range(10) :
        print i
    return df
        # time.sleep(1)
dview=c[:]
# print [i for i in range(10)]
dview.scatter('df', [i for i in range(11)])
res= c[:].apply_sync(runx)

add = lambda a,b: a+b
U = reduce(add, [1,2,3],100)
print U
# V = reduce(add, (r[1] for r in res))/k

# exit()
import numpy as np
from numpy.linalg import norm
# a=np.array([1 for i in range(9)]).reshape(3,3)
a=np.array([[1,2,3],[1,2,3]])
# print a
print norm(a)
import math
print math.sqrt(14)
print math.sqrt(28)

# print enumerate([1,2,3,4])
for count in enumerate([1,2,3,4]):
    print count