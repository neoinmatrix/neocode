import numpy as np , timeit 
def DFT_slow(x):  
    x = np.asarray(x, dtype=float)  
    # print x
    N = x.shape[0]  
    # print x.size
    n = np.arange(N)  
    # print n
    k = n.reshape((N, 1))  
    # print k
    M = np.exp(-2j * np.pi * k * n / N)  
    # print np.exp(k*n)
    # print M
    return np.dot(M, x) 

def FFT(x):  
    """A recursive implementation of the 1D Cooley-Tukey FFT"""  
    x=np.asarray(x,dtype=float)  
    N=x.shape[0]  
    if N%2>0:  
        raiseValueError("size of x must be a power of 2")  
    elif N<=4:  # this cutoff should be optimized  
        return DFT_slow(x)  
    else:  
        X_even=FFT(x[::2])  
        X_odd=FFT(x[1::2])  
        factor=np.exp(-2j*np.pi*np.arange(N)/N)  
        return np.concatenate([  X_even+factor[:N/2]*X_odd , X_even+factor[N/2:]*X_odd ])  

x=np.random.random(1024) 
%timeit FFT(x)
# a=np.arange(10)
# print a[::2]
# print a[1:7:2]
# print np.allclose(DFT_slow(x),np.fft.fft(x))  
# print x
# print DFT_slow(x)
# print np.fft.fft(x)
# %timeit DFT_slow(x)  
# %timeit np.fft.fft(x) 
