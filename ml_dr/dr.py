# coding=utf-8
# neomatrix want to learn and solve the digit recogniztion
# 2017-02-23
from numpy import * 
import time
class DigitR:
	# init data 
	def __init__(self):
		self.structData()

	# load data from file
	def loadData(self,path):
		file=open(path)
		tmp_data=[]
		for tmp in file:
			if tmp.find('pix')>=0:
				continue
			tmp_data.append([int(i) for i in tmp.split(',')])
		return tmp_data

	# normalize the data 
	def normalize(self,data):
		tmp_arr=[]
		for line in data:
			tmp_arr.append([1 if i>0 else 0 for i in line])
		return tmp_arr

	# struct the train and test data
	def  structData(self):
		data_tmp=array(self.loadData("./small_train.csv"))
		label_data=data_tmp[:,0]
		train_data=data_tmp[:,1:]
		# data_tmp=array(self.loadData("./test.csv"))
		# label_data_real=data_tmp[:,0]
		# test_data=data_tmp[:,1:]
		test_data=array(self.loadData("./small_test.csv"))

		self.label_data=label_data
		# self.label_data_real=label_data_real
		self.train_data=self.normalize(train_data)
		self.test_data=self.normalize(test_data)

	# view the character
	def viewCharacter(self,data,width=28,height=28):
		for i in range(height):
			tmp=""
			for j in range(width):
				tmp+=str(data[i*28+j])
			print tmp

	# calc the distance between two data 
	def calcDistance(self,da,db):
		distance=0
		for i in range(len(da)):
			distance+= 0 if (da[i]-db[i])==0 else 1
		return distance

	# get most likely number
	def getKNN(self,testa,k=3):
		distances=[]
		for i in range(len(self.train_data)):
			distances.append((i,self.calcDistance(testa,self.train_data[i])))
		# sort by distance
		distances.sort(key=lambda t: t[1],reverse=False)
		labels=[i[0] for i in distances[0:k]]
		nums=self.label_data[labels]
		count=[0 for i in range(10)]
		# find k near the most real data
		for i in nums:
			count[i]+=1
		# print labels
		# print distances[:k]
		# print self.label_data[labels]
		return count.index(max(count))

	def testData(self):
		start = time.clock()
		file=open("result.csv","w+")
		num=len(self.test_data)
		count=0
		file.write("ImageId,Label\n")
		for i in range(num):
			file.write(str(i+1)+","+str(self.getKNN(self.test_data[i]))+"\n")
			print str(i+1)," ",self.getKNN(self.test_data[i])
			if count%100 ==0 :
				end = time.clock()
				print "read: %f s" % (end - start) 
			count+=1
		file.close()
		print "ok"
		# 	if self.label_data_real[i]==self.getKNN(self.test_data[i]):
		# 		count+=1
		# print num," : ",count/num


if __name__ =="__main__":
	dr=DigitR()
	dr.testData()
	# dr.viewCharacter(dr.test_data[211])
	# print dr.getKNN(dr.test_data[211])

	# a=array([[1,2],[3,4]])
	# b=mat(a)
	# print a[0,:]
	# print b[0,:]

