# coding=utf-8
for i in range(1,10):
	tmp_str=""
	for j in range(1,10):
		tmp_str+=str(i)+"*"+str(j)+"="+str(i*j)+" "
	print tmp_str