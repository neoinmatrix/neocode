# coding=utf-8
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

data=load_iris()
features=data['data']
# print features
feature_names=data['feature_names']
# print feature_names
target=data['target']
# print len(target)
# for t,marker,c in zip(xrange(3),">ox","rgb"):
# 	print t,marker,c

# draw picture ===============================================
# for t,marker,c in zip(xrange(3),">ox","rgb"):
# 	plt.scatter(features[target==t,0],features[target==t,2],marker=marker,c=c)
# plt.show()
# print type(features)

# classify the data
plength=features[:,2]
is_setosa=(target==0)
# print plength
# print is_setosa
max_setosa=plength[is_setosa].max()
min_non_setosa=plength[~is_setosa].min()
# print max_setosa,min_non_setosa
# print format(max_setosa)
# print type(features)
# if features[:,2].all()<2:
# 	print 'Iris Setosa'
# else:
# 	print 'Iris Virginica or Iris Versicolour'
# for i in range(len(features[:,2])):
# 	if features[i,2]<2:
# 		print 'Iris Setosa'
# 	else:
# 		print 'Iris Virginica or Iris Versicolour'
features=features[~is_setosa]
# print features
# print target[~is_setosa]
labels=labels[~is_setosa]