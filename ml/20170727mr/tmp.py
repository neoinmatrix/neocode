

strx=''
for i in range(1,101):
    strx+='%.4d H \n'%i

with open('./tmp.txt','w') as f:
    f.write(strx)
print "ok"