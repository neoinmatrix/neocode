# coding=utf-8
import numpy as np

class DataSet:
    train_file_path='./data/dsjtzs_txfz_training.txt'
    test_file_path='./data/dsjtzs_txfz_test1.txt'
    train={"state":False}
    test={"state":False}

    def __init__(self,trainfp='',testfp=''):
        if  trainfp!='':
            self.train_file_path=trainfp
        if  testfp!='':
            self.test_file_path=testfp

    def __del__(self): 
        if self.test["state"]==True:
            self.test["file"].close()

    def getTrainData(self):
        idxs=[]
        labels=[]
        goals=[]
        mouses=[]
        with open(self.train_file_path,'r') as f:
            line='1'
            while line:
                line=f.readline()
                if line=='':
                    break
                idx,mouse,goal,label=self.dealLine(line)
                idxs.append(idx)
                mouses.append(mouse)
                goals.append(goal)
                labels.append(label)

        self.train["idxs"]=np.array(idxs)
        self.train["labels"]=np.array(labels)
        self.train["goals"]=np.array(goals)
        self.train["mouses"]=mouses
        self.train["size"]=len(idxs)
        return self.train

    def getPosOfMouse(self,idx=0):
        mouses=self.train["mouses"]
        mouse_arr=[]
        for v in mouses:
            if idx>=len(v[0]):
                mouse_arr.append([0,0,0])
            else:
                mouse_arr.append([v[0][idx],v[1][idx],v[2][idx]])
        return np.mat(mouse_arr)

    def dealMouse(self,mouse):
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
        if len(x_arr)==0:
            return np.zeros([3,1])
        x_arr=np.array(x_arr)
        y_arr=np.array(y_arr)
        t_arr=np.array(t_arr)
        return np.array([x_arr,y_arr,t_arr])

    def dealGoal(self,goal):
        garr=goal.split(',')
        if len(garr)!=2:
            return np.array([0,0])
        return np.array([float(garr[0]),float(garr[1])])

    def dealLine(self,line):
        linecols=line.split(' ')
        idx=int(linecols[0])
        mouse=self.dealMouse(linecols[1])
        goal=self.dealGoal(linecols[2])
        if len(linecols)==4:
            label=int(linecols[3])
        else:
            label=-1
        return idx,mouse,goal,label

    def readTestFile(self):
        if self.test["state"]==False:
            f=open(self.test_file_path,'r')
            self.test["state"]=True
            self.test["file"]=f
        f=self.test["file"]
        line=f.readline()
        if line=='':
            self.test["state"]=False
            self.test["file"]=''
            f.close()
            return False,False,False,False
        idx,mouse,goal,label=self.dealLine(line)
        return idx,mouse,goal,label


def main():
    ds=DataSet()
    ds.getTrainData()
    print ds.readTestFile()
    print ds.readTestFile()
    # ds.getTrainData()
    # print ds.train["size"]

if __name__=='__main__':
    main()

