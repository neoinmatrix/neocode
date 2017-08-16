import numpy as np
from sklearn.linear_model import LinearRegression
a=np.array([[1,1],[2,4],[3,9]])
b=np.array([1,2,3])
lr=LinearRegression()
lr.fit(a,b[:,np.newaxis])
print lr
print lr.coef_[0]
print b
# dw=datadraw.DataDraw('2d')
# x=mouses[1][0]
# t=mouses[1][2]
# plt.plot(t,x,c='r')
# # print x
# # print t
# tt= np.linspace(0,t[-1],15)
# # print tt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn import linear_model
# poly_reg = PolynomialFeatures(degree = 4) 
# X_poly = poly_reg.fit_transform(t[:,np.newaxis])
# print X_poly
# exit()
# ttp = poly_reg.fit_transform(tt[:,np.newaxis])
# lin_reg_2 = linear_model.LinearRegression()
# lin_reg_2.fit(X_poly, x[:,np.newaxis])
# rr=lin_reg_2.predict(ttp)