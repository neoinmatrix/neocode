# coding=utf-8
# this is script to solve the Titanic problem
# writen by neomatrix 2017-02-24

import pandas as pd
import numpy as np
import math

class Titanic:

	def __init__(self,selement=[]):
		self.train=pd.read_csv("./train.csv")
		self.test=pd.read_csv("./test.csv")
		# self.test=self.train[:][801:]
		# self.train=self.train[:][:801]
		# print self.test["Name"]
		if len(selement)==0: #SibSp	Parch 'Cabin','SibSp','Parch' Embarked
			self.selement=['Pclass','Sex','Age','Fare']
			# self.selement_rate=[float(i) for i in range(len(self.selement))]
		else:
			self.selement=selement
			self.selement_rate=[float(i) for i in range(len(selement))]
		self.category={}
		self.test_tmp={}
		
		for selement in self.selement:
			self.category.setdefault(selement,[{},{}])
			self.test_tmp.setdefault(selement,{})
		self.categoryData()

	# calc the survived or died the rate
	def categoryData(self):
		for selement in self.selement:
			data=self.normalize(selement,self.train[selement])
			rates=self.classify(data,1) # Survived
			self.category[selement][1]=rates
			rates=self.classify(data,0) # Died
			self.category[selement][0]=rates
	
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
					tmp.append(-1)
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
			else:
				tmp.append(data[i])
		return tmp

	# calc the rates
	def classify(self,data,Survived=1):
		length=len(data)
		if length==0:
			return None
		counts={}
		for i in range(length):
			columnValue=data[i]
			counts.setdefault(columnValue, 0)
			if self.train['Survived'][i]==Survived: # Survived
				counts[columnValue] += 1
		# change to rate
		rates={}
		for col in counts:
			rates.setdefault(col, 0)
			rates[col]=float(counts[col])/float(length)
		return rates

	# cal the one rate
	def calcRate(self,i):
		survived=1.0
		died=1.0
		# for selement in self.selement:
		# 	classify=self.test_tmp[selement][i]
		# 	died*=self.category[selement][0][classify]
		# 	survived*=self.category[selement][1][classify]
		for ix in range(len(self.selement)):
			selement=self.selement[ix]
			# rate=self.selement_rate[ix]
			classify=self.test_tmp[selement][i]
			died*=self.category[selement][0][classify]
			survived*=self.category[selement][1][classify]

		return survived,died

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


	# test my data 
	def testData(self):
		for selement in self.selement:
			data=np.array(self.test[selement])
			data=self.normalize(selement,data)
			self.test_tmp[selement]=data
		length=len(self.test['PassengerId'])

		passengers=np.array(self.test["PassengerId"])
		# passengers_survived=np.array(self.test["Survived"])
		tmp="PassengerId,Survived\n"
		for i in range(length):
			survived,died=self.calcRate(i)
			survived_flag=False
			if survived>died:
				tmp+=str(passengers[i])+",1\n"
			else:
				tmp+=str(passengers[i])+",0\n"
		file=open('./result.csv','w')
		file.write(tmp)
		
	# calc the correct rate of result
	def resultRate(self):
		right=pd.read_csv("./gender_submission.csv")
		result=pd.read_csv("./result.csv")
		# print result
		length=len(right["Survived"])
		count=0
		for i in range(length):
			if right["Survived"][i]==result["Survived"][i]:
				# print i,"ok"
				count+=1
		print length," | ",count," | ",(float(count)/float(length))
		
if __name__=='__main__':
	tnc=Titanic()
	# print tnc.category
	# tnc.testData()
	tnc.resultRate()

