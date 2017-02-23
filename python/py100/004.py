# coding=utf-8
	
# import time
# dtstr = str(raw_input('Enter the datetime:(20151215):'))
dtstr = '20170126'
dt = datetime.datetime.strptime(dtstr, "%Y%m%d")
another_dtstr =dtstr[:4] +'0101'
# print type(dt)
another_dt = datetime.datetime.strptime(another_dtstr, "%Y%m%d")
print (int((dt-another_dt).days) + 1)