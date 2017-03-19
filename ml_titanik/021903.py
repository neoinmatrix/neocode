# coding=utf-8
import pandas as pd
# print pd.get_dummies(['ab','b','c','a'])
a = pd.read_csv("./result.csv") 
b = pd.read_csv("./titanic.csv")
# print a
# print b
for i in range(len(a["PassengerId"])):
    print a["PassengerId"][i],a["Survived"][i],b["Survived"][i]