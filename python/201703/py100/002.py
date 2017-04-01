# coding=utf-8
# profit = int(raw_input('Enter the profit:'))
profit=    2000000
peroids = [1000000,600000,400000,200000,100000,0]
# peroids=[0,100000,200000,400000,600000,1000000]
rat = [0.01,0.015,0.03,0.05,0.075,0.1]
# rat=[0.1,0.075,0.05,0.03,0.015,0.01]

sum_memoy=0.0
for p in range(6):
	if profit>peroids[p]:
		sum_memoy+=(profit-peroids[p])*rat[p]
print sum_memoy
# sum_memoy=0.0
# # print peroids[-1]
# print range(1,6)
# for p in range(1,6):
# 	if p==5 and profit>peroids[p]:
# 		sum_memoy+=(profit-peroids[p])*rat[p]
# 	if profit<=peroids[p]:
# 		sum_memoy+=(profit-peroids[p-1])*rat[p]
# 		break
# 	else:
# 		sum_memoy+=(peroids[p]-peroids[p-1])*rat[p]
# print sum_memoy