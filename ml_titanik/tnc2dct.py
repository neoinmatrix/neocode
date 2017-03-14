# coding=utf-8
# this is script to solve the Titanic problem
# writen by neomatrix 2017-02-24

import pandas as pd
import numpy as np
import math

from DescisionTree import *

class Titanic_decision_tree:

	def __init__(self,selement=[]):
		self.train=pd.read_csv("./train.csv")
		self.test=pd.read_csv("./test.csv")
		if len(selement)==0: #SibSp	Parch 'Cabin','SibSp','Parch' Embarked
			self.selement=['Pclass','Sex','Age','Fare','Parch']
			self.selement_rate=np.array([1.0,1.0,1.0,1.0,1.0])
			self.selement_rate/=len(self.selement_rate)
		else:
			self.selement=selement
			self.selement_rate=[float(i) for i in range(len(selement))]
		self.category={}
		self.test_tmp={}
		# print self.selement_rate
		# print self.test.description()
		# print self.test.describe()
		# for selement in self.selement:
		# 	self.category.setdefault(selement,[{},{}])
		# 	self.test_tmp.setdefault(selement,{})
		# standordize the data
		for selement in self.selement:
			tmp=self.normalize(selement,self.train[selement])
			# tmp=np.array(tmp,np.float)
			# tmp-=tmp.min(0)
			# tmp/=(tmp.max(0)-tmp.min(0))
			self.train[selement]=tmp
			tmp=self.normalize(selement,self.test[selement])
			# tmp=np.array(tmp,np.float)
			# tmp-=tmp.min(0)
			# tmp/=(tmp.max(0)-tmp.min(0))
			self.test[selement]=tmp
		# print self.train["Pclass"]
		tmp_train=[]
		for i in range(len(self.train["PassengerId"])):
			tmp=[]
			for selement in self.selement:
				tmp.append(self.train[selement][i])
			tmp.append(self.train["Survived"][i])
			tmp_train.append(tmp)
		self.train=tmp_train

		tmp_test=[]
		PassengerId=[]
		for i in range(len(self.test["PassengerId"])):
			tmp=[]
			for selement in self.selement:
				tmp.append(self.test[selement][i])
			# tmp.append(self.train["Survived"][i])
			tmp_test.append(tmp)
			PassengerId.append(self.test["PassengerId"][i])
		self.test=tmp_test
		self.passengerid=PassengerId
		# print tmp_train
	
	# normalize the data
	def normalize(self,selement,data):
		tmp=[]
		for i in range(len(data)):
			# elif selement=='Fare':
			# 		tmp.append(int(data[i]/10))
			if selement=='Sex':
				if data[i]=='female':
					tmp.append(0)
				else:
					tmp.append(1)
			elif selement=='Age':
				if math.isnan(data[i])==False:
					tmp.append(int(data[i]/22))
				else:
					tmp.append(4)
			elif selement=='Fare':
					if math.isnan(data[i])==False:
						tmp.append(int(data[i]/10))
					else:
						tmp.append(0)
			elif selement=='Embarked':
				if data[i]=='C':
					tmp.append(1)
				elif data[i]=='Q':
					tmp.append(2)
				elif data[i]=='S':
					tmp.append(3)
				else:
					tmp.append(-1)
			elif selement=='Cabin':
					if type(data[i]) is float:
						tmp.append(1)
					else:
						tmp.append(0)
			elif selement=='Parch':
				if data[i]>2:
					tmp.append(2)
				else:
					tmp.append(data[i])
			elif selement=='Name':
				if data[i].find('Miss')>-1:
					tmp.append(1)
				elif data[i].find('Mrs')>-1:
					tmp.append(2)
				else:
					tmp.append(3)
			else:
				tmp.append(data[i])
		return tmp

count=0
def classify(inputTree, featLabels, testVec):  
    firstStr = inputTree.keys()[0]  
    # print firstStr
    # return 
    # print featLabels
    # return 
    secondDict = inputTree[firstStr]  
    featIndex = featLabels.index(firstStr)  
    # return 
    # global count
    # count+=1
    classLabel=secondDict["default"]
    for key in secondDict.keys():  
        if testVec[featIndex] == key:  
            if type(secondDict[key]).__name__ == 'dict': 
                
                # if count==4:
                #     print  secondDict[key]
                #     print featLabels
                #     print testVec
                #     exit()
                # print count
                # if key in secondDict
                classLabel = classify(secondDict[key], featLabels, testVec)  
            else: 
            	classLabel = secondDict[key]  
    return classLabel 

if __name__=='__main__':
	tnc=Titanic_decision_tree()
	labels=tnc.selement[:]
	myTree = createTree(tnc.train,labels) 
	labels=tnc.selement[:]
	# print myTree 
	# print classify(myTree,labels,tnc.test[5])
	# print tnc.train
	# print tnc.test
	# print tnc.test[0]
	# print classify(myTree,labels,tnc.test[0])

	result=pd.read_csv("./result.csv")
	# print result

	count=0
	tmp="PassengerId,Survived\n"
	for i in range(len(tnc.passengerid)):
		survived=classify(myTree,labels,tnc.test[i])
		tmp+="%d,%d\n"%(tnc.passengerid[i],survived)
		# print tnc.passengerid[i],survived
		if result["Survived"][i]==survived:
			count+=1
	print float(count)/float(len(tnc.passengerid))

	# print tmp
	file=open('./result_dt.csv','w')
	file.write(tmp)

	# print tnc.passengerid