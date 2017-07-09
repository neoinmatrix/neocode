# coding=utf-8
import numpy as np 

# the start point of tracking has influenced classifier
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
        xstd=data[:,0].std()
        ymin=data[:,1].min()
        ymax=data[:,1].max()
        ystd=data[:,1].std()
        tmin=data[:,2].min()
        tmax=data[:,2].max()
        tstd=data[:,2].std()
        return [xmin-xstd,xmax+xstd,ymin-ystd,ymax+ystd,tmin-tstd,tmax+tstd]
    mouses_start=get_start(mouses)
    borders=[[0]*6]*3
    borders[0]=calc_borders(mouses_start[2600:2800])
    borders[1]=calc_borders(mouses_start[2800:2900])
    borders[2]=calc_borders(mouses_start[2900:3000])
    return np.array(borders)

def get_spoint_filter(mouses,config):
    borders=config["borders"]
    x_start=mouses[0][0]
    y_start=mouses[1][0]
    t_start=mouses[2][0]
    flag=1.0
    # the start point is not in shape is manual tracking
    for i in range(3):
        xmin,xmax,ymin,ymax,tmin,tmax=borders[i]
        if x_start>=xmin and x_start<=xmax:
            if y_start>=ymin and y_start<=ymax:
                # if t_start>=tmin and t_start<=tmax:
                    flag=0.0
                    break
    return [flag]

# if x toward change must be manual tracking 
# x change then t change so ingore t change
def get_X_PN(mouse):
    x=mouse[0]
    t=mouse[2]
    n=len(x)
    flag=0.0
    for i in range(1,n):
        if x[i-1]>x[i]:
            flag=1.0
            break
    return [flag]

# print demo
def printdemo(vector):
    np.set_printoptions(formatter={'float':lambda x: "%5.2f"%float(x)})
    # print vector[0:10]
    # print vector[:,0].min()
    # print vector[:,0].max()
    # print vector[:,0].mean()
    # print vector[:,0].std()
    # exit()
    # for i in vector[0:2600]:
    #     if i>0:
    #     print i 
    print  vector[2000:2600]
    # print  vector[2600:3000]
    print  vector[0:2600].mean()
    print  vector[2600:3000].mean()
    # print vector[2600:3000]
    # print vector
    # print vector[1000:1010]
    # print "=============="
    # # print vector[2700:2710]
    # print vector[2800:2810]
    exit()
    scaler = preprocessing.StandardScaler().fit(X)
    for i in range(3):
        a=vector[0:2600,i].mean()
        b=vector[2600:3000,i].mean()

        ar=vector[0:2600:,i].std()
        br=vector[2600:3000:,i].std()
        print a-ar," ",a+ar
        print b-br," ",b+br

        # print vector[0:2600,i].mean()
        # print vector[2600:3000,i].mean()
        # print vector[0:2600:,i].std()
        # print vector[2600:3000:,i].std()
        print i,"============"
    exit()
