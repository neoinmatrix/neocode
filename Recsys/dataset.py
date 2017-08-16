# coding=utf-8
import re
import random
import numpy as np

class Wsdata:
    def __init__(self,path=''):
        if path!='':
            self.path=path
        else:
            self.path='/home/neo/source/source/wsdream/'

    def getRtMatrix(self):     
        data=np.loadtxt(self.path+'rtMatrix.txt')
        # print data.shape
        # print data.min()
        # print data.max()
        # print data.mean()
        # print data.std()
        self.rt_matrix=data   
        # # (339, 5825)
        # # -1.0
        # # 19.999
        # # 0.811108740932
        # # 1.9670462271 
    
    def getTpMatrix(self):     
        data=np.loadtxt(self.path+'tpMatrix.txt')
        # print data.shape
        # print data.min()
        # print data.max()
        # print data.mean()
        # print data.std()
        self.tp_matrix=data
        # # (339, 5825)
        # # -1.0
        # # 1000.0
        # # 44.0346226863
        # # 107.439394348

    def getUserList(self):    
        with open(self.path+'userlist.txt') as f:
            header=f.readline()
            headers=re.findall(r'\[(.*?)\]',header)
            self.user_headers=headers
            f.readline()
            content=f.read()
            items=re.findall(r'(.*?)\n',content)
            userlist=[]
            for item in items:
                if item=='':
                    continue
                tmp_arr=re.findall(r'(.*?)\t',item)
                userlist.append(tmp_arr)
            self.user_list=userlist

    def getWsList(self):    
        with open(self.path+'wslist.txt') as f:
            header=f.readline()
            headers=re.findall(r'\[(.*?)\]',header)
            self.ws_headers=headers
            f.readline()
            content=f.read()
            items=re.findall(r'(.*?)\n',content)
            wslist=[]
            for item in items:
                if item=='':
                    continue
                tmp_arr=re.findall(r'(.*?)\t',item)
                wslist.append(tmp_arr)
            self.ws_list=wslist    

    def sampling(self,item,percent=0.1):
        # mean=item.mean()
        # std=item.std()
        n=len(item)
        idxs=random.sample(range(n), int(n*percent))
        idxs=sorted(idxs)
        idxs_test=[]
        for i in range(n):
            if i not in idxs:
                idxs_test.append(i)
        return idxs,idxs_test

    def getSample(self,type='rt',percent=0.1):
        if type=='rt':
            self.getRtMatrix()
            data=self.rt_matrix
        else:
            self.getTpMatrix()
            data=self.tp_matrix
        n,m=data.shape
        # sample_data=[]
        # sample_idxs=[]  
        # sample_data=[]
        # sample_idxs=[]
        str_data=''
        str_data_test=''
        for i in range(n):
            s_idxs,s_idxs_test=self.sampling(data[i],percent=0.1)
            # sample_idxs.append(s_idxs)
            # sample_data.append(data[i,s_idxs])

            # sample_idxs.append(s_idxs)
            # sample_data.append(data[i,s_idxs])
            for j in s_idxs:
                str_data+="%d\t%d\t%f\n"%(i,j,data[i,j])    
            for j in s_idxs_test:
                str_data_test+="%d\t%d\t%f\n"%(i,j,data[i,j])
        path=self.path+'sample/sample_%s_%2.1f.txt'%(type,percent)
        with open(path,'w') as f:
            f.write(str_data) 
        path=self.path+'sample/test_%s_%2.1f.txt'%(type,percent)
        with open(path,'w') as f:
            f.write(str_data_test)
        print path+" written is ok "
        # sample_idxs=np.array(sample_idxs)
        # print sample_idxs.shape

    def getRtData(self,file=''):
        if file=='':
            file='sample/sample_rt_0.1.txt'
        path=self.path+file
        with open(path,'r') as f:
            content=f.read()
        items_arr=content.split('\n')
        items=[]
        n_client=0
        n_service=0
        for v in items_arr:
            if v=='':
                continue
            tmp_arr=v.split('\t')
            items.append((int(tmp_arr[0]),int(tmp_arr[1]),float(tmp_arr[2])))
            if int(tmp_arr[0])>n_client:
                n_client=int(tmp_arr[0])
            if int(tmp_arr[1])>n_service:
                n_service=int(tmp_arr[1])
        # print n_client,n_service
        print items[0]


if __name__=="__main__":
    import dataset
    wsdata=dataset.Wsdata()
    # wsdata.getRtMatrix()
    # wsdata.getTpMatrix()
    # wsdata.getUserList()
    # wsdata.getWsList()
    # print "data is ready!"
    wsdata.getSample('rt')
    
    # wsdata.getSample('tp')
    # wsdata.getRtData()
    # a=np.ones(100)
    # wsdata.sampling(a,0.1)
