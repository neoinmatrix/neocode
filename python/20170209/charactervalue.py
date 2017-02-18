import numpy as np

x= np.mat([
	[0.65,0.28,0.7],
	[0.15,0.67,0.18],
	[0.12,0.36,0.52]])
a,b=np.linalg.eig(x)
print a
print b
sum= a[0]+ a[1]+ a[2]
print a[0]/sum
print a[1]/sum
print a[2]/sum