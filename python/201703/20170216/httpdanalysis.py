# coding=utf-8
import re
import sys
import os

path = './20170215.log'
if len(sys.argv)>1:
	input_string=sys.argv[1]
	if input_string=="-h" or input_string=="--h" or input_string=="help":
		print "input the xxx.log name to analyst the data"
		sys.exit(0)
	else:
		if os.path.exists(sys.argv[1]):
			path=sys.argv[1]
print "time           time_wait close_wait established threads"
# TIME_WAIT 2210
# CLOSE_WAIT 86
# SYN_SENT 1
# FIN_WAIT2 28
# ESTABLISHED 245
# SYN_RECV 5
file=open(path,'r')
lines = file.read(-1)
items=lines.split('-------------------')
def getParams(data):
	tmp_lines=data.split("\n")
	params={"TIME_WAIT":"0","CLOSE_WAIT":"0","SYN_SENT":"0","FIN_WAIT1":"0","FIN_WAIT2":"0","ESTABLISHED":"0","SYN_RECV":"0"}
	for v in tmp_lines:
		tmp_data=v.split(" ")
		if len(tmp_data)!=2:
			continue
		if tmp_data[1].find(":")>0:
			params["time"]=tmp_data[0]+" "+tmp_data[1]
		else:
			params[tmp_data[0]]=tmp_data[1]
	return params

data=[]
for v in items:
    tmp=v.split('===================')
    if len(tmp)!=2:
        continue
    params=getParams(tmp[0])
    stuff = re.findall('[0-9]+',tmp[1])
    threads=int(stuff[0])*int(stuff[1])+int(stuff[2])*int(stuff[3])
    data.append([params["time"],params["TIME_WAIT"],params["CLOSE_WAIT"],params["ESTABLISHED"],threads])

for v in data:
    tmp=""
    for k in v:
        tmp+=str(k).ljust(5)+" |"
    print tmp
 

	

# str='''Fri Feb 10 20:56:01 CST 2017
# TIME_WAIT 2210
# CLOSE_WAIT 86
# SYN_SENT 1
# FIN_WAIT2 28
# ESTABLISHED 245
# SYN_RECV 5
# ===================
#      |-httpd-+-16*[httpd---65*[{httpd}]]
#      |-16*[httpd---65*[{httpd}]]'''

# tmp=str.split('===================')
# tmp2=tmp[0].split("\n")
# # print tmp[0]
# params={}
# for v in tmp2:
# 	tmp_data=v.split(" ")
# 	if len(tmp_data)!=2:
# 		continue
# 	params[tmp_data[0]]	=tmp_data[1]

# print params
