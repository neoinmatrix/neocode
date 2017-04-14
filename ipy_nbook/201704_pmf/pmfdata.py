# coding=utf-8
import numpy as np
data=np.ones([10,10])*np.nan

for i in range(0,10):
	for j in range(0,10):
		for k in range(np.random.randint(2)):
			data[i,j]=np.random.randint(5)*1.0+1.0

tmp_data=[]
for i in range(0,10):
	for j in range(0,10):
		if np.isnan(data[i,j])==False:
			tmp_data.append([i,j,data[i,j]])
file_str=""
for i in range(0,10):
	for j in range(0,10):
		if np.isnan(data[i,j])==False:
			file_str+="%d,%d,%f\n"%(i,j,data[i,j])
# print file_str
file=open("./create.csv","w")
file.write(file_str)
file.close()
print "ok"
# print tmp_data
# print len(tmp_data)
# data.to_csv('create.csv')