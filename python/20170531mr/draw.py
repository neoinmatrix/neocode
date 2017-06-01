# coding=utf-8
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  

labels=[0]*3000
goals=['']*3000
mouses=[['']]*3000
speeds=[0.0]*3000

def draw(data,fname='./a.png',save=False):
    x=data[0]
    y=data[1]
    plt.plot(x,y)
    if save:
        plt.savefig(fname)
        plt.clf()
        plt.close()
    else:
        plt.show()

def posdeal(mouse):
    marr=mouse.split(';')
    x_arr=[]
    y_arr=[]
    t_arr=[]
    tt=0
    for v in marr:
        varr=v.split(',')
        if len(varr)!=3:
            continue
        x=int(varr[0])
        y=int(varr[1])
        t=int(varr[2])
        x_arr.append(x)
        y_arr.append(y)
        t_arr.append(t/100)
    return [x_arr,y_arr,t_arr]

def datadeal():
    with open('./dsjtzs_txfz_training.txt','r') as f:
        line='1'
        while line:
            line=f.readline()
            if line=='':
                break
            linecols=line.split(' ')
            idx=int(linecols[0])-1
            mouse=linecols[1]
            goal=linecols[2]
            label=int(linecols[3])

            labels[idx]=label
            goals[idx]=goal
            mouses[idx]=mouse
            # marr=mouse.split(';')
            # speeds[idx]=calc(mouse)
    # print speeds

def draw3d(data,ax):
    X=data[0]
    Y=data[1]
    Z=data[2]

 
    # X = [1, 1, 2, 2]  
    # Y = [3, 4, 4, 3]  
    # Z = [1, 2, 1, 1]  
    # ax.scatter(X, Y, Z) 
    ax.plot(X,Y,Z)
    # plt.show()
    # plt.clf()
    # plt.close()  
def drawgoal(data,ax):
    ax.scatter(data[0],data[1],data[2])

datadeal()
# print posdeal(mouses[0])
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d') 

draw3d(posdeal(mouses[0]),ax)

# arr=goals[0].split(',')
# drawgoal([float(arr[0]),float(arr[1]),0],ax)
# print arr
draw3d(posdeal(mouses[2798]),ax)
# arr=goals[2798].split(',')

# drawgoal([float(arr[0]),float(arr[1]),0],ax)
# print posdeal(mouses[2798])
plt.show()
# for i in range(len(mouses)):
#     fname="./pic/%.3d.png"%(i+1)
#     draw(posdeal(mouses[i]),fname,True)
#     print i
# print "ok";
# mpos1=posdeal(mouses[0])
# draw(mpos1,save=True)
# print mpos1