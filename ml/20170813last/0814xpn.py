import sys
from pyspark import SparkContext
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing

output_path = sys.argv[1]
input_train = sys.argv[2]
input_all = sys.argv[3]

sc = SparkContext(appName="train")
rdd_train = sc.textFile(input_train)
rdd_all = sc.textFile(input_all)

# common func =========================
def spt_mouse(raw):
    items=raw.split(' ')
    mtr_x=[]
    mtr_y=[]
    mtr_t=[]
    for v in items[1].split(';'):
        tmp=v.split(',')
        if len(tmp)!=3:
            continue
        mtr_x.append(float(tmp[0]))
        mtr_y.append(float(tmp[1]))
        mtr_t.append(float(tmp[2]))
    gtmp=items[2].split(',')
    goal=[float(gtmp[0]),float(gtmp[1])]
    if len(items)==4:
        label=int(items[3])
    else:
        label=0
    return [int(items[0]),np.array([mtr_x,mtr_y,mtr_t]),goal,label]

# def flt_mac(mouse):
#     x=mouse[0]
#     n=len(x)
#     count=0
#     for i in range(1,n):
#         if x[i-1]>x[i]:
#             count+=1
#         if float(count)/float(n)>0.15: 
#             return False
#     return True

def flt_mac(mouse):
    x=mouse[0]
    n=len(x)
    count=0
    for i in range(1,n):
        if x[i-1]>x[i]:
            count+=1
    if count==0:
        return False
    p=float(count)/float(n)
    if p>0 and p<0.1: 
        if np.random.randint(1000)==333:
            return True
    return False
    

 
output = rdd_all.map(spt_mouse).filter(lambda d:flt_mac(d[1])).map(lambda d:d[1])
num=output.collect()
print " classifier results:",num
# def get_nd(d,idxx):
#     n=len(d)
#     for i in range(len(idxx)):
#         if idxx[i]<0:
#             idxx[i]=0
#         if idxx[i]>(n-1):
#             idxx[i]=n-1
#     d=np.array(d)
#     return d[idxx].flatten()


# # training the data
# def get_trains():
#     trains = rdd_train.map(spt_mouse).filter(lambda d:flt_mac(d[1])).map(get_feats)
#     trains =trains.collect()
#     vtrs={"sharp":[],"diff":[],"middle":[],"all":[]}
#     lbs={"sharp":[],"diff":[],"middle":[],"all":[]}
#     for d in trains:
#         idx=d[0]
#         label=d[3]
#         f={"sharp":4,"diff":5,"middle":6,"all":7}
#         for k,v in f.items():
#             f[k]=d[v]

#         if idx in range(2601):
#             for k in vtrs:
#                 vtrs[k].append(f[k])
#                 lbs[k].append(label)
#             continue
#         k=""
#         if idx in range(2901,3001):
#             k="sharp"
#             vtrs[k].append(f[k])
#             lbs[k].append(label)
#         if idx in range(2701,2901):
#             k="diff"
#             vtrs[k].append(f[k])
#             lbs[k].append(label)
#         if idx in range(2601,2801):
#             k="middle"
#             vtrs[k].append(f[k])
#             lbs[k].append(label)
#         if idx in range(2601,3001):
#             k="all"
#             vtrs[k].append(f[k])
#             lbs[k].append(label)
        
#     scalar={"sharp":[],"diff":[],"middle":[],"all":[]}
#     clf={"sharp":[],"diff":[],"middle":[],"all":[]}
#     # training 
#     for k in scalar:
#         vector=np.array(vtrs[k])
#         label=np.array(lbs[k])
#         scalar[k]= preprocessing.StandardScaler().fit(vector)
#         vector = preprocessing.scale(vector)    
#         if k=='all':
#             clf[k]= SVC(C=2.5,kernel='linear')
#         else:
#             clf[k]= SVC(C=1.5)

#         clf[k].fit(vector,label)
    
#     return clf,scalar