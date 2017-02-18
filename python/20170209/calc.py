from math import log
p=[0.4,0.3,0.15,0.05,0.04,0.03,0.03]
sum=0.0
for k in p:
	# print k
	sum+=k*log(k)/log(2)
print -1*sum
