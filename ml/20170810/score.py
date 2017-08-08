# 61.21
import numpy as np
np.set_printoptions(formatter={'float':lambda x: "%10.2f"%float(x)})

m=188008.0
s=0.6044
n=np.array(range(10*10000,40*10000+10000,10000),dtype=np.float)
x=s*0.2*(2*n+3*m)

# print n
# print x

m=185597
s=0.6121
n=np.array(range(10*10000,40*10000+10000,10000),dtype=np.float)
x=s*0.2*(2*n+3*m)
# print n
# print x

mh=185597.0
sh=0.6121

ml=188008.0
sl=0.6044
n=1.5*(sh*mh-sl*ml)/(sl-sh)
print n


# m=110598.0
# s=0.4246
# n=np.array(range(10*10000,40*10000+10000,10000),dtype=np.float)
# x=s*0.2*(2*n+3*m)

# print n
# print x

# mh=110598.0
# sh=0.4246

# ml=110696.0
# sl=0.4224
# n=1.5*(sh*mh-sl*ml)/(sl-sh)
# print n

# the evaluated about 15w items with machine 
# 
# [ 100000.00  110000.00  120000.00  130000.00  140000.00  150000.00
#   160000.00  170000.00  180000.00  190000.00  200000.00]
# [  45159.95   46858.35   48556.75   50255.15   51953.55   53651.95
#    55350.35   57048.75   58747.15   60445.55   62143.95]

# print "========================"
# m=290598.0
# s=0.228
# n=np.array(range(10*10000,40*10000+10000,10000),dtype=np.float)
# x=s*0.2*(2*n+3*m)

# print n
# print x

# m=158043.0
# s=0.4295
# n=np.array(range(10*10000,40*10000+10000,10000),dtype=np.float)
# x=s*0.2*(2*n+3*m)

# print n
# print x

