from time import time
import numpy as np 

def rander(seed,n):
    if n ==0:
        return 0
    seed = int(seed)
    length = len(str(seed))
    seed = int(seed**2/pow(10,(length/2))) % int(pow(10.0,length))
    # print(str(seed) + "",end="")
    # print seed
    # rander(seed,n-1)

def rand(a=0,b=0,seed=0):
    if seed==0:
        seed = int(time())
    length = len(str(seed))
    seed = int(seed**2/pow(10,(length/2))) % int(pow(10.0,length))
    if a!=0 and b==0:
        b=a
        a=0
    if b==0:
        b=1
    return seed%b+a


def main():
    seed=time()
    a=[]
    for i in range(2000):
        # print rand(0,10,seed+i*10)
        a.append(rand(0,10,seed+i*11))
    a=np.array(a)
    # print a
    import matplotlib.pyplot as plt
    plt.hist(a)
    plt.show()

main()