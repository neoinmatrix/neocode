# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

def plotnumber(train,index):
    # print train.iloc[index,0]
    img=train.iloc[index,1:].values.reshape(28,28)
    savepath="%d_%d.txt"%(index,train.iloc[index,0])
    np.savetxt(savepath,np.ceil(img/127),fmt="%d")

    plt.imshow(img)
    plt.show()

def drawnumber(start,dims):
    n=dims[0]
    m=dims[1]
    for i in range(n):
        for j in range(m):
            idx=i*m+j+1
            plt.subplot(n,m,idx)
            img=train.iloc[idx,1:].values.reshape(28,28)
            plt.imshow(img)
    plt.show()

def draw(X,Y,newX):
    plt.scatter(X[0,:],X[1,:])
    plt.scatter(Y/(2**0.5),[0]*Y.shape[1],c='r',marker='<')
    plt.scatter(newX[0,:],newX[1,:],c='y',marker='x')

    plt.grid(True)
    plt.plot([-2.5,0.5,1,2.5], [-2.5,0.5,1,2.5], 'g-')
    plt.axis([-2.5,2.5,-2.5 ,2.5])
    plt.yticks([i for i in range(-3,4,1)])
    plt.xticks([i for i in range(-3,4,1)])
    plt.show()

def drawpcadata(start,dims):
    n=dims[0]
    m=dims[1]
    pca = decomposition.PCA()
    for i in range(n):
        for j in range(m):
            idx=i*m+j+1
            plt.subplot(n,m,idx)
            img=train.iloc[idx,1:].values.reshape(28,28)
            img=img/128
            pca.fit(img)
            pca_result=pca.transform(img)
            plt.imshow(pca_result)
    plt.show()

if __name__=="__main__":
    train=pd.read_csv("mnist.csv")
    # plotnumber(train,11)
    # drawnumber(0,[5,5])
    drawpcadata(0,[2,2])


    # img=train.iloc[2,1:].values.reshape(28,28)
    # img=img/128
    # # print img 
    # # img=img.reshape([1,len(img)])
    # # print img
    # pca = decomposition.PCA()
    # print pca.fit(img)
    # pca_result=pca.transform(img)

    # pcadata=[0.0]*28
    # for i in range(28):
    #     pcadata[i]=pca_result[:,i].sum()
    # pcadata=sorted(pcadata)
    # pcadata=pcadata[-1::-1]
    # print pcadata
    # # print pcadata
    # plt.imshow(pca_result)
    # # plt.plot([i for i in range(1,29)],pcadata)
    # plt.show()
    # print pca_result.shape
    # # train_pca = pd.DataFrame()

