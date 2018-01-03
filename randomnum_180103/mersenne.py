from time import time
import numpy as np 

index = 624
MT = [0]*index
# MT[0] ->seed

def inter(t):
    return(0xFFFFFFFF & t)

def twister():
    global index
    for i in range(624):
        y = inter((MT[i] & 0x80000000) +(MT[(i + 1) % 624] & 0x7fffffff))
        MT[i] = MT[(i + 397) % 624] ^ y >> 1
        if y % 2 != 0:
            MT[i] = MT[i] ^ 0x9908b0df
    index = 0

def exnum():
    global index
    if index >= 624:
        twister()
    y = MT[index]
    y = y ^ y >> 11
    y = y ^ y << 7 & 2636928640
    y = y ^ y << 15 & 4022730752
    y = y ^ y >> 18
    index = index + 1
    return inter(y)

def mainset(seed):
    MT[0] = seed    #seed
    for i in range(1,624):
        MT[i] = inter(1812433253 * (MT[i - 1] ^ MT[i - 1] >> 30) + i)
    return exnum()

def main():
    mi = 1
    ma = 100    
    so = mainset(int(time())) / (2**32-1)
    rd = mi + int((ma-mi)*so)
    print rd 



# def main():
#     seed=time()
#     a=[]
#     mi = 0
#     ma = 10
#     for i in range(200):
#         rd = LCG(seed+i*11)
#         ourd = int((ma-mi)*rd) + mi
#         a.append(ourd)
#     a=np.array(a)
#     # print a
#     import matplotlib.pyplot as plt
#     plt.hist(a)
#     plt.show()

main()