import numpy as np 
from data0812 import * 
# print len(sample)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import pylab

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




for i in range(0,300):
    draw_analyst_single(i,sample[i],0,0,path='./data/pic/',save=True)