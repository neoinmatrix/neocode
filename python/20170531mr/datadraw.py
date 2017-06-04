# coding=utf-8
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
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
        plt.scatter(data[0],data[1])

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

if __name__=="__main__":
   # draw3d()
   # draw2d()
   drawScatter()