path="./d.csv"
file=open(path,'r')
lines = file.read(-1)
data=lines.split("\n")
for v in data:
	tmp=v.split(" ,")
	if len(tmp)!=2:
		print tmp
		continue
	# print tmp[0],tmp[1]
	if tmp[0]!=tmp[1]:
		print tmp[0],tmp[1]
# print data