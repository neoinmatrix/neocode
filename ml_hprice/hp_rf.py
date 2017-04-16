# coding=utf-8

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

train=pd.read_csv("./input/train.csv")
train=train.fillna(0)
test=pd.read_csv("./input/test.csv")
test=test.fillna(0)
params=[]
for i in train.columns[train.columns!="SalePrice"]:
    if type(train[i][0])==np.int64 or  type(train[i][0])==np.float64:
        tmp=train["SalePrice"].corr(train[i])
        if np.fabs(tmp)>0.26:
            params.append(i)

x_train=train[params].values
y_train=train['SalePrice'].values
x_test=test[params].values

rf=RandomForestRegressor(max_depth=20, random_state=2)
rf.fit(x_train, y_train)
predicted= rf.predict(x_test)
# expected=y_test
# sumx=0.0
# for i in range(0,len(expected)):
#     sumx+=((expected[i]-predicted[i])/100000)**2
# print np.sqrt(sumx/len(expected))
# #     print expected[i],predicted[i]
# print mean_squared_error(expected/100000,predicted/100000)

tmpstr="Id,SalePrice\n"
for i in range(len(test["Id"])):
    tmpstr+="%d,%f\n"%(test["Id"][i],np.fabs(predicted[i]))

file=open('./result.csv','w')
file.write(tmpstr)
file.close()
