# coding=utf-8
import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import  metrics
from datadeal import DataDeal
 
dd=DataDeal()
# data,target,nums,types
pima=dd.getPima()
iris=dd.getIris()

# def getPR(data):
    
def common(clf,clf_name=''):
    print "========",clf_name,"========"
    # pima
    data,target,nums,types=pima
    result=dd.test(clf,data,target,nums,types,isprint=False)
    print "=====pima====",result[1]
    # iris
    data,target,nums,types=iris
    result=dd.test(clf,data,target,nums,types,isprint=False)
    print "=====iris====",result[1] 
# knn =====================================
def knn():
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    common(clf,"knn")
# naive_bayes ===================================
def bayes():
    from sklearn.naive_bayes import GaussianNB 
    clf = GaussianNB()
    common(clf,"knn")
# DecisionTree ===================================
def dtree():
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(max_depth=5)
    common(clf,"DecisionTree")
# RandomForest ===================================
def randforest():
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=4, n_estimators=20)
    common(clf,"RandomForest")
# ann ===================================
def ann():
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(10,),max_iter=500, random_state=0)
    common(clf,"neural_network")
# svm ===================================
def svm():
    from sklearn.svm import SVC
    clf = SVC(C=1.6,gamma=0.6)
    common(clf,"svm c=1.6 rbf")
    clf = SVC(C=1.5, kernel='linear' )
    common(clf,"svm c=1.5 linear")
    clf = SVC(C=1, kernel='poly',degree=3,gamma=0.9 ,coef0=2)
    common(clf,"svm c=1 poly")
    clf = SVC(C=1, kernel='sigmoid',gamma=0.5,coef0=0)
    common(clf,"svm c=1 sigmoid")
# ada boost ===================================
def adaboost():
    from sklearn import tree
    t = tree.DecisionTreeClassifier(max_depth=7)
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(t)
    common(clf,"ada boost base on svm")

if __name__=="__main__":
    # knn()
    # bayes()
    # dtree()
    # randforest()
    # ann()
    # svm()
    adaboost()