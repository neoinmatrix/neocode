# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

clf = linear_model.LinearRegression()
# x=np.random.random((50))
# y=np.random.random((50))
# z=x+y
# intmp=[[tx,ty] for tx,ty in zip(x,y)]
# # print intmp
# clf.fit(np.array(intmp),z)
# print clf.coef_
# print clf.predict([[1.0,1.0]])
# z=x+y
# print z
train=pd.read_csv("./train.csv")
train=train.fillna(0.0)
result=[]
result_title=[]
for i in train.columns[train.columns!="SalePrice"]:
    if type(train[i][0])==np.int64 or  type(train[i][0])==np.float64:
        tmp=train["SalePrice"].corr(train[i])
        result.append(tmp)
        result_title.append(i)
        if tmp>0.1:
            result.append(tmp)
            result_title.append(i)
# for i in result_title:
#     min=train[i].min()
#     max=train[i].max()
#     train[i]=(train[i]-min)/(max-min)
# traint=train
# train=traint.ix[[i for i in range(0,1000)]]
# test=traint.ix[[i for i in range(1000,1460)]]

clf.fit(train[result_title].values,train["SalePrice"].values)
# # print train.info()
# # print test.head()
test=pd.read_csv("./test.csv")
test=test.fillna(0.0)
# # print test.ix[[0,1]]
# # print test.ix[[1000,1001]][result_title]
# arr=[1000,1002,1003,1004]
result_predict=clf.predict(test[result_title])
# print test.head()
tmpstr="Id,SalePrice\n"
for i in range(len(test["Id"])):
    tmpstr+="%d,%f\n"%(test["Id"][i],np.fabs(result_predict[i]))
# print tmpstr
file=open('./result.csv','w')
file.write(tmpstr)
file.close()
print "ok"


# accuary=test["SalePrice"]
# tmp=np.fabs(result_predict-accuary).sum()
# print tmp
# print type(result_predict)
# print type(accuary.values)
# print result_predict
# print test.ix[arr]["SalePrice"]
# print clf.coef_
# print result_title
# plt.barh([i for i in range(len(result))],np.array(result))
# plt.yticks([float(i)+0.4 for i in range(len(result))],result_title)
# plt.show()
