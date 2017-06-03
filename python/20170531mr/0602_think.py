# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

labels=[0]*3000
goals=['']*3000
mouses=[['']]*3000
speeds=[0.0]*3000

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
            speeds[idx]=calc(mouse)
    print speeds

def datadealsingle():
    with open('./data/dsjtzs_txfz_training.txt','r') as f:
        line=f.readline()
        linecols=line.split(' ')
        idx=int(linecols[0])-1
        mouse=linecols[1]
        goal=linecols[2]
        label=int(linecols[3])

        labels[idx]=label
        goals[idx]=goal
        mouses[idx]=mouse
        # marr=mouse.split(';')
        speeds[idx]=calc(mouse.split(';'))
        return mouse.split(';')

def draw(data):
    x_arr=data[0]
    y_arr=data[1]
    plt.plot(x_arr,y_arr)
    plt.show()

def dealpos(mouse):
    m_x=[]
    m_y=[]
    mouse_arr=mouse.split(';')
    for v in mouse_arr:
        v_arr=v.split(',')
        if len(v_arr)!=3:
            continue
        x=int(v_arr[0])
        y=int(v_arr[1])
        m_x.append(x)
        m_y.append(y)
    return [m_x,m_y]

if __name__=='__main__':
    datadealsingle()
    # datadeal()
    datadealsingle()
    # pos1=dealpos(mouses[0])
    # pos2=dealpos(mouses[2998])
    # draw(pos1)
    # draw(pos2)

#TODO1: draw the picture of mouse path
#TODO2: analyst the data
#TODO3: the first and last drive the speed
#TODO4: fast to find the result
