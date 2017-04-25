# coding=utf-8
import numpy as np

def stochasticGradientDescent(X, Y, alpha, maxIterations=200):
    n,m=X.shape
    theta=np.random.rand(m)
    # theta=np.ones(m)
    index_arr=np.arange(n)
    for t in range(0, maxIterations):  
        np.random.shuffle(index_arr)
        old_theta=theta
        for i in index_arr:         #  ( h(Xi)-Yi ) * Xij    at i change theta all
            hypothesis = np.dot(X[i], theta)
            loss = hypothesis - Y[i]
            gradient = loss*X[i]
            theta = theta - alpha * gradient
        # print sum(np.fabs(old_theta-theta))
        if sum(np.fabs(old_theta-theta))<0.01:
            break
    return theta

X=np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
],dtype=np.float32)
Y=np.array([6,15,24],np.float32)
print stochasticGradientDescent(X,Y,0.005,100)
