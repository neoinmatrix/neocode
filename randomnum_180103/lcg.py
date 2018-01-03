from time import time
import numpy as np 


#define
m = 2**32
a = 1103515245
c = 12345

def LCG(seed):
    seed = (a * seed + c) % m
    return seed/float(m-1)

def main():
    seed=time()
    a=[]
    mi = 0
    ma = 10
    for i in range(200):
        rd = LCG(seed+i*11)
        ourd = int((ma-mi)*rd) + mi
        a.append(ourd)
    a=np.array(a)
    # print a
    import matplotlib.pyplot as plt
    plt.hist(a)
    plt.show()

main()