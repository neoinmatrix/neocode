# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train=pd.read_csv("mnist.csv")
def plotnumber(train,index):
    # print train.iloc[index,0]
    img=train.iloc[index,1:].values.reshape(28,28)
    savepath="%d_%d.txt"%(index,train.iloc[index,0])
    np.savetxt(savepath,np.ceil(img/127),fmt="%d")

    plt.imshow(img)
    plt.show()

# plotnumber(train,11)

def drawnumber(start,len):
    for i in range(start,start+len):
        plt.subplot(1,len,i+1)
        img=train.iloc[i,1:].values.reshape(28,28)
        plt.imshow(img)
    plt.show()
drawnumber(0,5)
