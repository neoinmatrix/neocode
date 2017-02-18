from math import log
from pylab import *  
p=arange(0.01,1,0.01)
h=[]
for v in p:
	h.append(-1*v*log(v))

plot(p, h) 
xlabel('x')  
ylabel('sin(x)')  

title('Simple plot')  
grid(True)  
show() 
# print p
# print h
# sum=0.0
# for k in p:
# 	# print k
# 	sum+=k*log(k)/log(2)
# print -1*sum