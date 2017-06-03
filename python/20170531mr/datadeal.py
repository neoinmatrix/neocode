# coding=utf-8
import numpy as np

labels=np.zeros((3000,1))
goals=['']*3000
mouses_raw=['']*3000
speeds=[0.0]*3000
mouses=['']*3000


def calc(mouse):
    marr=mouse.split(';')
    starts=marr[0].split(',')
    ends=marr[-2].split(',')
    sx=float(starts[0])
    sy=float(starts[1])
    st=float(starts[2])
    ex=float(ends[0])
    ey=float(ends[1])
    et=float(ends[2])
    length=((sx-ex)**2+(sy-ey)**2)**0.5
    elapse=et-st
    if elapse<1e-10:
        return 1000000
    else:
        return length/elapse

def getdata():
    with open('./data/dsjtzs_txfz_training.txt','r') as f:
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
            mouses_raw[idx]=mouse
            # marr=mouse.split(';')
            # speeds[idx]=calc(mouse)
    # print speeds

def posdata(mouse):
    marr=mouse.split(';')
    x_arr=[]
    y_arr=[]
    t_arr=[]
    for v in marr:
        varr=v.split(',')
        if len(varr)!=3:
            continue
        x=float(varr[0])
        y=float(varr[1])
        t=float(varr[2])
        x_arr.append(x)
        y_arr.append(y)
        t_arr.append(t)

    x_arr=np.array(x_arr)
    y_arr=np.array(y_arr)
    t_arr=np.array(t_arr)
    # if x_arr.max()!=x_arr.min():
    #     x_arr=(x_arr-x_arr.min())/(x_arr.max()-x_arr.min())
    # else:
    #     x_arr=(x_arr-x_arr.min())/x_arr.max()
    # if y_arr.max()!=y_arr.min():
    #     y_arr=(y_arr-y_arr.min())/(y_arr.max()-y_arr.min())
    # else:
    #     y_arr=(y_arr-y_arr.min())/y_arr.max()
    # if t_arr.max()!=t_arr.min():
    #     t_arr=(t_arr-t_arr.min())/(t_arr.max()-t_arr.min())
    # else:
    #     t_arr=(t_arr-t_arr.min())/t_arr.max()
    return np.array([x_arr,y_arr,t_arr])

def fatdata():
    for i in range(len(mouses_raw)):
        mouses[i]=posdata(mouses_raw[i])

def initdata():
    getdata()
    fatdata()


def main():
    # print mouses_raw[1]
    # print posdata(mouses_raw[492])
    pass 
    # print labels
    # print posdata(mouses_raw[0])
    # print mouses[0][0]
    # datadeal()

if __name__=='__main__':
    main()
    pass


#TODO1: draw the picture of mouse path
#TODO2: analyst the data
#TODO3: the first and last drive the speed
#TODO4: fast to find the result
