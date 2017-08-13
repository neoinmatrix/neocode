import numpy as np
# from sdata import *
# from sdata2 import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import pylab

import dataset
import datadeal
import datadraw


def draw_analyst_single(idx,mouse,goal,label,notfind='',nmachine='',path='',stop=1000,save=False):
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3dview:NO.%.4d'%idx,loc='left')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')
    ax.plot(mouse[0],mouse[1],mouse[2])
    ax.scatter(mouse[0],mouse[1],mouse[2])

    ax = fig.add_subplot(332)
    # fig.set_alpha(0.1)
    ax.set_title('xy')
    ax.plot(mouse[0],mouse[1],c='r')
    ax.scatter(mouse[0],mouse[1],c='r')

    ax = fig.add_subplot(334)
    ax.set_title('xt')
    ax.plot(mouse[0],mouse[2],c='g')
    ax.scatter(mouse[0],mouse[2],c='g')

    ax = fig.add_subplot(336)
    ax.set_title('yt')
    ax.plot(mouse[1],mouse[2],c='b')
    ax.scatter(mouse[1],mouse[2],c='b')

    if save==True:
        print idx
        plt.title(str(idx))
        plt.savefig(path+"%.4d.png"%idx)
        plt.clf()
        plt.close()
    else:
        plt.show()

def drawold():
    count=1
    for mouse in sample:
        draw_analyst_single(count,mouse,0,0,path='./data/pic2/',save=True)
        count+=1
# draw_analyst_single(count,sample[0],0,0,path='./data/pic/',save=False)

# def draw_train()

def draw_end_target(mouse,goal,lastn=5,save=False,c='r'):
    x=mouse[0]
    y=mouse[1]
    t=mouse[2]
    n=len(x)
    # plt.plot(x[n-lastn:n],y[n-lastn:n],c='r')
    plt.plot([x[-1],goal[0]],[y[-1],goal[1]],c=c)
    plt.scatter([goal[0]],[goal[1]],c='b')



    # if save==True:
    #     print idx
    #     plt.title(str(idx))
    #     plt.savefig(path+"%.4d.png"%idx)
    #     plt.clf()
    #     plt.close()
    # else:
    #     plt.show()

ds=dataset.DataSet()
ds.getTrainData()
mouses=ds.train["mouses"]
goals=ds.train["goals"]
labels=ds.train["labels"]
n=ds.train["size"]

for i in range(2600):
    if i%26!=23:
        continue
    if i in range(2600,2700):
        c='b'
    elif i in range(2700,2800):
        c='g'
    elif i in range(2800,2900):
        c='y'
    else:
        c='r'
    draw_end_target(mouses[i],goals[i],lastn=2,c=c)

for i in range(2700,2800):
    # if i%26!=23:
    #     continue
    if i in range(2600,2700):
        c='b'
    elif i in range(2700,2800):
        c='g'
    elif i in range(2800,2900):
        c='y'
    else:
        c='r'
    draw_end_target(mouses[i],goals[i],lastn=2,c=c)
plt.show()
# x1,y1=mouses[0][-2],mouses[1][-2] 
# x2,y2=mouses[0][-1],mouses[1][-1] 

# print x1
# plt.plot([x1,x2],[y1,y2],c='r')
# plt.scatter([goals[0][0]],[goals[0][1]],c='b')
# plt.show()
# for i in range(1,n+1):
#     draw_analyst_single(i,mouses[i],0,0,path='./data/trains/',save=True)



    