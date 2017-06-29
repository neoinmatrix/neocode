# coding=utf-8
import dataset
import datadeal
import datadraw
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import  metrics
import matplotlib.pyplot as plt 


def analyst_xyt():
    def getxyt(idx,mouse,goal,label):
        tmp=[]
        xlen=len(mouse[0])
        tmp.append(xlen) # len 
        for i in range(3):
            tmp.append(mouse[i].min()) # x min
            tmp.append(mouse[i].max()) # x max
            tmp.append(mouse[i].mean()) # x max
            tmp.append(mouse[i].std()) # x max
        return np.array(tmp).reshape([1,len(tmp)])
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    for i in range(n):
        vector.append(getxyt(1,mouses[i],goals[i],1)[0])
    vector=np.array(vector)

    name=["mouse_size",\
    "x_min","x_max","x_mean","x_std",\
    "y_min","y_max","y_mean","y_std",\
    "t_min","t_max","t_mean","t_std",\
    ]
    allx='name\tmin\tmax\tmean\tstd\n'
    
    def getanalyst(start,end):
        result=''
        for j in range(13):
            size=[]
            for i in range(start,end):
                idx=3
                size.append(vector[i,j])
            size=np.array(size)
            result+="%s\t%10.5f\t%10.5f\t%10.5f\t%10.5f"%(name[j],size.min(),size.max(),size.mean(),size.std())
            result+="\n"
        return result


    # allx+=getanalyst(0,2600)
    # allx+='-\t-\t-\t-\t-\n'
    # allx+=getanalyst(2600,3000)
    # with open('./data/xyt/xyt.csv','w') as f:
    #     f.write(allx)
    # print allx
    def draw():
        import matplotlib.pyplot as plt
        for j in range(13):
            size=[]
            for i in range(n):
                idx=3
                size.append(vector[i,j])
            size=np.array(size)
            # result+="%s\t%10.5f\t%10.5f\t%10.5f\t%10.5f"%(name[j],size.min(),size.max(),size.mean(),size.std())
            # result+="\n"        
            plt.plot(range(3000),size)
            plt.title(name[j])
            plt.savefig('./data/xyt/%s.png'%name[j])
            plt.clf()
            # plt.show()
            # exit()
            print j
        # plt.close(0)
        # plt.show()
    draw()

def analyst_xyt2():
    def getxyt(idx,mouse,goal,label):
        tmp=[]
        tmp.append(mouse[0][0])
        tmp.append(mouse[1][0])
        tmp.append(mouse[2][-1])
        # xlen=len(mouse[0])
        # tmp.append(xlen) # len 
        # for i in range(3):
        #     tmp.append(mouse[i].min()) # x min
        #     tmp.append(mouse[i].max()) # x max
        #     tmp.append(mouse[i].mean()) # x max
        #     tmp.append(mouse[i].std()) # x max
        return np.array(tmp).reshape([1,len(tmp)])
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    labels=ds.train["labels"]
    n=ds.train["size"]
    vector=[]
    for i in range(n):
        vector.append(getxyt(1,mouses[i],goals[i],1)[0])
    vector=np.array(vector)

    name=["mouse_size",\
    "x_min","x_max","x_mean","x_std",\
    "y_min","y_max","y_mean","y_std",\
    "t_min","t_max","t_mean","t_std",\
    ]
    allx='name\tmin\tmax\tmean\tstd\n'
    
    def getanalyst(start,end):
        result=''
        for j in range(13):
            size=[]
            for i in range(start,end):
                idx=3
                size.append(vector[i,j])
            size=np.array(size)
            result+="%s\t%10.5f\t%10.5f\t%10.5f\t%10.5f"%(name[j],size.min(),size.max(),size.mean(),size.std())
            result+="\n"
        return result


    # allx+=getanalyst(0,2600)
    # allx+='-\t-\t-\t-\t-\n'
    # allx+=getanalyst(2600,3000)
    # with open('./data/xyt/xyt.csv','w') as f:
    #     f.write(allx)
    # print allx
    def draw(j):
        import matplotlib.pyplot as plt
        size=[]
        for i in range(n):
            size.append(vector[i,j])
        size=np.array(size)    
        plt.plot(range(3000),size)
        plt.show()
        # plt.title(name[j])
        # plt.savefig('./data/xyt/%s.png'%name[j])
        # plt.clf()
    draw(2)

if __name__=="__main__":
    # analyst_xyt()
    analyst_xyt2()