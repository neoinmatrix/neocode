# coding=utf-8
import numpy as np

def batchGradientDescent(X, Y, alpha, maxIterations=200):
    n,m=X.shape
    theta=np.random.rand(m)
    # theta=np.ones(m)
    XT = X.transpose()
    for t in range(0, maxIterations):  # sum(i:0->m)[ ( h(Xi)-Yi ) * Xij ]/m
        hypothesis = np.dot(X, theta)
        loss = hypothesis - Y
        gradient = np.dot(XT, loss) / m
        theta = theta - alpha * gradient
        if np.abs(loss.sum())<0.01:
            break
    return theta

X=np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
],dtype=np.float32)
Y=np.array([6,15,24],np.float32)
print batchGradientDescent(X,Y,0.005,100)
