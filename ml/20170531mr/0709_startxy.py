# coding=utf-8
import datadraw
import dataset
import datadeal

import numpy as np 
import matplotlib.pyplot as plt

def get_start(data):
    x_arr=[]
    y_arr=[]
    t_arr=[]
    for v in data:
        x=v[0][0]
        x_arr.append(x)
        y=v[1][0]
        y_arr.append(y)
        t=v[2][0]
        t_arr.append(t)
    return np.array([x_arr,y_arr,t_arr]).T

def main_start():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]

    dw=datadraw.DataDraw("2d")
    # mouses_start=ds.getPosOfMouse(0)
    # mouses_start=mouses_start.T
    # print mouses_start.shape
    # exit()
    # print mouses_start[:2600].shape
    # print mouses_start[:,0].shape
    # exit()
    # print mouses_start[:,0].T.shape
    # mouses_start[:,0].T
    mouses_start=get_start(mouses)
    # print data.shape 
    # plt.scatter([1,2],[2,3],c='y')
    # print mouses_start[:10]
    # exit()
    path="./data/start_pic/"
    for i in range(2650,3001,50):
        dw.drawbatchgoal(mouses_start[:2600],'y')
        dw.drawbatchgoal(mouses_start[i-50:i],'r')
        plt.title("%d.png"%i)
        plt.savefig(path+"%d.png"%i)
        plt.clf()
        plt.close()
        print i
    # dw.drawbatchgoal(mouses_start[2650:2850],'g')
    # dw.drawbatchgoal(mouses_start[2850:2950],'g')
    # dw.drawbatchgoal(mouses_start[2850:3000],'b')
    # dw.drawbatchgoal(mouses_start[2600:],'g')
    # dw.drawbatchgoal(mouses_start[2600:],'b')
    # ==    ==
    # dw.drawbatchgoal(goals[:2600],'y')
    # dw.drawbatchgoal(goals[2600:],'b')
    # plt.show()

def get_borders(mouses):
    def get_start(data):
        x_arr=[]
        y_arr=[]
        t_arr=[]
        for v in data:
            x=v[0][0]
            x_arr.append(x)
            y=v[1][0]
            y_arr.append(y)
            t=v[2][0]
            t_arr.append(t)
        return np.array([x_arr,y_arr,t_arr]).T
    def calc_borders(data):
        xmin=data[:,0].min()
        xmax=data[:,0].max()
        ymin=data[:,1].min()
        ymax=data[:,1].max()
        tmin=data[:,2].min()
        tmax=data[:,2].max()
        return [xmin,xmax,ymin,ymax,tmin,tmax]
    mouses_start=get_start(mouses)
    borders=[[0]*6]*3
    borders[0]=calc_borders(mouses_start[2600:2800])
    borders[1]=calc_borders(mouses_start[2800:2900])
    borders[3]=calc_borders(mouses_start[2900:3000])
    return np.array(borders)

def drawrectangle(data):
    xa,xb,ya,yb=data
    x=[xa,xa,xb,xb,xa]
    y=[ya,yb,yb,ya,ya]
    plt.plot(x,y)

def main2d():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    dw=datadraw.DataDraw("2d")
    mouses_start=get_start(mouses)
    # print mouses_start.shape
    
    dw.drawbatchgoal(mouses_start[0:2600],'y')
    dw.drawbatchgoal(mouses_start[2650:2850],'g')
    dw.drawbatchgoal(mouses_start[2850:2950],'g')
    dw.drawbatchgoal(mouses_start[2850:3000],'y')
    dw.drawbatchgoal(mouses_start[2600:],'g')
    dw.drawbatchgoal(mouses_start[2600:],'b')

    borders=get_borders(mouses)
    for i in range(4):
        drawrectangle(borders[i])

    # print borders
    plt.show()
    # print borders

def drawbatch(data,dw,c):
    x=data[0]
    y=data[1]
    t=data[2]
    n=len(x)
    for i in range(n):
        dw.draw3dgoal([x[i],y[i],t[i]],c=c)

def main():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    dw=datadraw.DataDraw("3d")
    mouses_start=get_start(mouses)
    # print mouses_start.shape
    # dw.draw3dgoal

    # drawbatch(mouses_start[0:2600].T,dw,'y')

    drawbatch(mouses_start[2650:2800].T,dw,'g')
    drawbatch(mouses_start[2800:2900].T,dw,'r')
    drawbatch(mouses_start[2900:3000].T,dw,'b')

    # dw.draw3dgoal(mouses_start[0:2600],c='y')
    # dw.draw3dgoal(mouses_start[2650:2850],c='g')
    # dw.draw3dgoal(mouses_start[2850:2950],c='g')
    # dw.draw3dgoal(mouses_start[2850:3000],c='y')
    # dw.draw3dgoal(mouses_start[2600:],c='g')
    # dw.draw3dgoal(mouses_start[2600:],c='b')

    # borders=get_borders(mouses)
    # for i in range(4):
    #     drawrectangle(borders[i])

    # print borders
    plt.show()
    # print borders
       
if __name__=="__main__":
    import time
    start =time.clock()
    main()
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))


