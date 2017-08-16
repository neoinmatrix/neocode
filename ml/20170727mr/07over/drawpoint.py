import sys
from pyspark import SparkContext
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing

output_path = sys.argv[1]
input_all = sys.argv[2]
# input_all = sys.argv[3]

sc = SparkContext(appName="train")
# rdd_train = sc.textFile(inputx)
rdd_all = sc.textFile(input_all)

#  data format =========================
def splitx(raw):
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

def get_X_PN_isMachine(mouse):
    x=mouse[0]
    n=len(x)
    for i in range(1,n):
        if x[i-1]>x[i]:
            return False
    return True
    
def filterXNP(data):
    pass
    return get_X_PN_isMachine(data[1])


def getStart(data):
    mouse=data[1]
    return [mouse[0][0],mouse[1][0]]

output = rdd_all.map(splitx).filter(filterXNP).map(getStart)

# num=output.count()
# print "l \\ shape numbers:",num

output.saveAsTextFile(output_path)