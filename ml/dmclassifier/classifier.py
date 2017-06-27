# coding=utf-8
import numpy as np 
from datadeal import DataDeal

# the class designed by myself
dd=DataDeal()
# data,target,nums,types
pima=dd.getPima()
iris=dd.getIris()
# just show the params
def common(clf,clf_name=''):
    print "========",clf_name,"========"
    # pima
    data,target,nums,types=pima
    result=dd.test(clf,data,target,nums,types,isprint=False)
    print "=====pima===="
    print "%5.3f %5.3f %5.3f %5.3f"%(result["acc"]/10.0,\
        result["p"]/10.0,result["r"]/10.0,result["f1"]/10.0)
    # iris
    data,target,nums,types=iris
    result=dd.test(clf,data,target,nums,types,isprint=False)
    print "=====iris===="
    print "%5.3f %5.3f %5.3f %5.3f"%(result["acc"]/10.0,\
        result["p"]/10.0,result["r"]/10.0,result["f1"]/10.0)
# show the params name
def common_human(clf,clf_name=''):
    print "========",clf_name,"========"
    # pima
    data,target,nums,types=pima
    result=dd.test(clf,data,target,nums,types,isprint=False)
    print "=====pima====\n"
    print "accuracy:%5.3f"%(result["acc"]/10.0)
    print "precision:%5.3f"%(result["p"]/10.0)
    print "recall:%5.3f"%(result["r"]/10.0)
    print "f1_score:%5.3f"%(result["f1"]/10.0)
    # iris
    data,target,nums,types=iris
    result=dd.test(clf,data,target,nums,types,isprint=False)
    print "=====iris====\n"
    print "accuracy:%5.3f"%(result["acc"]/10.0)
    print "precision:%5.3f"%(result["p"]/10.0)
    print "recall:%5.3f"%(result["r"]/10.0)
    print "f1_score:%5.3f"%(result["f1"]/10.0)
# knn =====================================
def knn():
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    common(clf,"knn")
# naive_bayes ===================================
def bayes():
    from sklearn.naive_bayes import GaussianNB 
    clf = GaussianNB()
    common(clf,"bayes")
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
    clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(5,),max_iter=1000, random_state=0)
    common(clf,"neural_network")
    clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(10,),max_iter=1000, random_state=0)
    common(clf,"neural_network")
    clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(15,),max_iter=1000, random_state=0)
    common(clf,"neural_network")
# svm ===================================
def svm():
    from sklearn.svm import SVC
    # clf = SVC(C=1.6,gamma=0.6)
    # common(clf,"svm c=1.6 rbf")
    # clf = SVC(C=1.5, kernel='linear' )
    # common(clf,"svm c=1.5 linear")
    # clf = SVC(C=1, kernel='poly',degree=3,gamma=0.9 ,coef0=2)
    # common(clf,"svm c=1 poly")
    clf = SVC(C=1.5, kernel='sigmoid',gamma=0.2,coef0=0)
    common(clf,"svm c=1 sigmoid")
# ada boost ===================================
def adaboost():
    from sklearn import tree
    t = tree.DecisionTreeClassifier(max_depth=6)
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(t)
    common(clf,"ada boost base on svm")
# draw the Correlation matrix between features
def drawCorrelation():
    # data,target,_,_=pima
    # data=data*10
    # data=np.c_[data,target*3]
    # _,n=data.shape
    # ref_matrix=np.cov(data.T)
    # labels=[i for i in range(n)]
    # name='Correlation'
    # title='Correlation (reflect rate between two features)'
    # dd.plot_correlation(ref_matrix,labels,name,title) 

    data,target,_,_=iris
    data=data*2.5
    data=np.c_[data,target*1.5]
    _,n=data.shape
    ref_matrix=np.cov(data.T)
    labels=[i for i in range(n)]
    name='Correlation'
    title='Correlation (reflect rate between two features)'
    dd.plot_correlation(ref_matrix,labels,name,title) 

if __name__=="__main__":
    # drawCorrelation()
    knn()
    # bayes()
    # dtree()
    # randforest()
    # ann()
    # svm()
    # adaboost()
    pass