# coding=utf-8
# this is script to solve the Titanic problem
# writen by neomatrix 2017-02-24

import pandas as pd
import numpy as np
import math
class Titanic_knn:

	def __init__(self,selement=[]):
		self.train=pd.read_csv("./train.csv")
		self.test=pd.read_csv("./test.csv")
		# self.test=self.train[:][801:]
		# self.train=self.train[:][:801]
		# print self.test["Name"]
		if len(selement)==0: #SibSp	Parch 'Cabin','SibSp','Parch' Embarked
			self.selement=['Pclass','Sex','Age','Fare','Name']
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
			tmp=np.array(tmp,np.float)
			tmp-=tmp.min(0)
			tmp/=(tmp.max(0)-tmp.min(0))
			self.train[selement]=tmp
			tmp=self.normalize(selement,self.test[selement])
			tmp=np.array(tmp,np.float)
			tmp-=tmp.min(0)
			tmp/=(tmp.max(0)-tmp.min(0))
			self.test[selement]=tmp
		# print self.train
	
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

	# test my data 
	def checkData(self):
		for selement in self.selement:
			data=np.array(self.test[selement])
			data=self.normalize(selement,data)
			self.test_tmp[selement]=data
		length=len(self.test['PassengerId'])

		passengers=np.array(self.test["PassengerId"])
		passengers_survived=np.array(self.test["Survived"])
		t2f=0
		f2t=0
		tt=0
		ff=0
		collect=""
		collectid=[]
		for i in range(length):
			survived,died=self.calcRate(i)
			tmp=''
			survived_flag=False
			if survived>died:
				survived_flag=True
				tmp+=' passenger '+str(passengers[i])+' survived'
			else:
				tmp+=' passenger '+str(passengers[i])+' died'
		
		
			if survived_flag==True and passengers_survived[i]==1:
				tmp+=" tt"
				tt+=1
			elif survived_flag==False and passengers_survived[i]==1:
				tmp+=" t2f"
				collect+=tmp+"\n"
				collectid.append(i)
				t2f+=1
			elif survived_flag==True and passengers_survived[i]==0:
				tmp+=" f2t"
				f2t+=1
			elif survived_flag==False and passengers_survived[i]==0:
				tmp+=" ff"
				ff+=1
			print tmp
		print tt," ",t2f," ",f2t," ",ff," "
		print "right rate:",float(tt+ff)/float(length)," "
		# print collect
		data=np.array(self.test)
		print data[0][:3],data[0][4:]
		for i in data[collectid]:
			print i[:3],i[4:]

	# # test my data 
	def testData(self):
		# for selement in self.selement:
		# 	data=np.array(self.test[selement])
		# 	data=self.normalize(selement,data)
		# 	self.test_tmp[selement]=data
		length=len(self.test['PassengerId'])

		passengers=np.array(self.test["PassengerId"])
		# passengers_survived=np.array(self.test["Survived"])
		tmp="PassengerId,Survived\n"
		for i in range(length):
			if self.getKNN(i)==1:
				print str(passengers[i])+",1"
				tmp+=str(passengers[i])+",1\n"
			else:
				print str(passengers[i])+",0"
				tmp+=str(passengers[i])+",0\n"

		# print tmp
			# survived,died=self.calcRate(i)
			# survived_flag=False
			# if survived>died:
			# 	tmp+=str(passengers[i])+",1\n"
			# else:
			# 	tmp+=str(passengers[i])+",0\n"
		file=open('./result2.csv','w')
		file.write(tmp)
		
	def calcDistance(self,idx,test_id):
		distances=0.0
		for i in range(len(self.selement)):
			selement=self.selement[i]
			tmp=self.train[selement][idx]-self.test[selement][test_id]
			distances=(tmp**2)*self.selement_rate[i]
			# print self.train[selement][idx],self.test[selement][test_id]

			# tmp= tmp if tmp>0 else -tmp
			# distances+=tmp
		return distances

	# get most likely number
	def getKNN(self,test_id,k=3):
		distances=[]
		for i in range(len(self.train)):
			distances.append((i,self.calcDistance(i,test_id)))
		# sort by distance
		distances.sort(key=lambda t: t[1],reverse=False)
		# labels=[i[0] for i in distances[0:k]]
		# print distances
		return self.train["Survived"][distances[0][0]]
		# nums=self.label_data[labels]
		# count=[0 for i in range(10)]
		# find k near the most real data
		# for i in nums:
		# 	count[i]+=1
		# print labels
		# print distances[:k]
		# print self.label_data[labels]
		# return count.index(max(count))

	#
	def checkResult(self):
		self.real=pd.read_csv("gender_submission.csv")
		self.test=pd.read_csv("./result2.csv")
		# print self.real
		# exit()
		t2f=0
		f2t=0
		tt=0
		ff=0
		collect=""
		collectid=[]
		length=len(self.real["PassengerId"])
		tmp=''
		for i in range(length):
			# print i
			if self.real["Survived"][i]==self.test["Survived"][i] and \
				self.real["Survived"][i]==1:
				tmp+=" tt"
				tt+=1
			elif self.real["Survived"][i]==self.test["Survived"][i] and \
				self.real["Survived"][i]==0:
				ff+=1
			elif self.real["Survived"][i]!=self.test["Survived"][i] and \
				self.real["Survived"][i]==1:
				t2f+=1
			elif self.real["Survived"][i]!=self.test["Survived"][i] and \
				self.real["Survived"][i]==0:
				f2t+=1
			# else:
			# 	# print "|",self.real["Survived"][i],"|"
			# 	print "|",self.test["Survived"][i],"|"

			# tmp=''
			# survived_flag=False
			# if survived>died:
			# 	survived_flag=True
			# 	tmp+=' passenger '+str(passengers[i])+' survived'
			# else:
			# 	tmp+=' passenger '+str(passengers[i])+' died'
		
		
			# if survived_flag==True and passengers_survived[i]==1:
			# 	tmp+=" tt"
			# 	tt+=1
			# elif survived_flag==False and passengers_survived[i]==1:
			# 	tmp+=" t2f"
			# 	collect+=tmp+"\n"
			# 	collectid.append(i)
			# 	t2f+=1
			# elif survived_flag==True and passengers_survived[i]==0:
			# 	tmp+=" f2t"
			# 	f2t+=1
			# elif survived_flag==False and passengers_survived[i]==0:
			# 	tmp+=" ff"
			# 	ff+=1
			# print tmp
		print tt," ",t2f," ",f2t," ",ff," "
		print "right rate:",float(tt+ff)/float(length)," "

if __name__=='__main__':
	tnc=Titanic_knn()
	tnc.checkResult()
	# tnc.testData()
	# print tnc.getKNN(3)
	# print tnc.calcDistance(1,2)
	
	# tnc.getKNN(1)
	# print tnc.category
	# tnc.testData()
	# tnc.resultRate()

