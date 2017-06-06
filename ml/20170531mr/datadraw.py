# coding=utf-8
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import pylab

import dataset
import datadraw

class DataDraw():
    ax=''
    def __init__(self,typex='3d'):
        if typex=="3d":
            fig = plt.figure()  
            self.ax = fig.add_subplot(111, projection='3d') 

    def drawline(self,data):
        x=data[0]
        y=data[1]
        plt.plot(x,y)

    def drawgoal(self,data,c='r'):
        plt.scatter(data[0],data[1],c=c)

    def drawbatchgoal(self,data,c='r'):
        plt.scatter(data[:,0],data[:,1],c=c)

    def draw(self,data,fname='./data/a.png',save=False):
        x=data[0]
        y=data[1]
        plt.plot(x,y)
        if save:
            plt.savefig(fname)
            plt.clf()
            plt.close()
        else:
            plt.show()

    def draw3dline(self,data,ax=''):
        if ax=='':
            ax=self.ax
        X=data[0]
        Y=data[1]
        Z=data[2]
        ax.plot(X,Y,Z)

    def draw3dgoal(self,data,ax='',c='r'):
        if ax=='':
            ax=self.ax
        ax.scatter(data[0],data[1],data[2],c=c)

def draw3d():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    dw=datadraw.DataDraw()

    START=2700
    PAIRS=2
    colors=['b','r','g','y','c','k','m']
    for i in range(PAIRS):
        dw.draw3dline(mouses[i])
        dw.draw3dline(mouses[START+i])
        dw.draw3dgoal([goals[i][0],goals[i][1],i],c=colors[i%7])
        dw.draw3dgoal([goals[START+i][0],goals[START+i][1],START+i],c=colors[(i+3)%7])
    plt.show()

def draw2d():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    dw=datadraw.DataDraw("2d")

    START=2700
    PAIRS=2
    colors=['b','r','g','y','c','k','m']
    for i in range(PAIRS):
        dw.drawline(mouses[i])
        dw.drawline(mouses[START+i])
        # dw.drawgoal([goals[i][0],goals[i][1],i],c=colors[i%7])
        # dw.drawgoal([goals[START+i][0],goals[START+i][1]],c=colors[(i+3)%7])
    plt.show()

def drawScatter():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    dw=datadraw.DataDraw("2d")
    mouses_start=ds.getPosOfMouse(1)
    dw.drawbatchgoal(mouses_start[:2600],'y')
    dw.drawbatchgoal(mouses_start[2600:],'b')

    dw.drawbatchgoal(goals[:2600],'y')
    dw.drawbatchgoal(goals[2600:],'b')
    
    plt.show()

def plot_confusion_matrix(cm, genre_list, name, title,max,save=False):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Greens', vmin=0, vmax=max)
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    cm=cm.T
    for i in range(len(cm)):
        # t=len(cm[0])-i
        for j in range(len(cm[0])):
            if cm[i,j]<1e-2:
                continue
            pylab.text(i, j, '%.2f'%cm[i,j])

    pylab.title(title)
    pylab.colorbar()
    pylab.grid(True)
    pylab.show()
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(True)
    if save==True:
        pylab.savefig(os.path.join(CHART_DIR, "confusion_matrix_%s.png"%name), bbox_inches="tight")


if __name__=="__main__":
   # draw3d()
   # draw2d()
   drawScatter()